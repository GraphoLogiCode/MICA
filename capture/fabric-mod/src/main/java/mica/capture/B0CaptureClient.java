package mica.capture;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonNull;
import com.google.gson.JsonObject;
import com.mojang.blaze3d.pipeline.RenderTarget;
import com.mojang.blaze3d.platform.InputConstants;
import com.mojang.blaze3d.platform.NativeImage;
import com.mojang.blaze3d.systems.RenderSystem;
import com.mojang.blaze3d.vertex.PoseStack;
import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientLifecycleEvents;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.keybinding.v1.KeyBindingHelper;
import net.fabricmc.fabric.api.client.rendering.v1.HudRenderCallback;
import net.fabricmc.fabric.api.event.player.PlayerBlockBreakEvents;
import net.fabricmc.fabric.api.event.player.UseBlockCallback;
import net.minecraft.client.KeyMapping;
import net.minecraft.client.Minecraft;
import net.minecraft.client.gui.Font;
import net.minecraft.client.player.LocalPlayer;
import net.minecraft.core.BlockPos;
import net.minecraft.core.Registry;
import net.minecraft.world.InteractionResult;
import net.minecraft.world.item.BlockItem;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.phys.BlockHitResult;
import net.minecraft.world.phys.EntityHitResult;
import net.minecraft.world.phys.HitResult;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.lwjgl.glfw.GLFW;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Queue;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

// Streams the client-side half of B0 in real time, and records it to disk.
//
// PRIMARY path = a live TCP stream to D1/D2 (see LiveStream below): every tick's packet
// goes straight out a localhost socket with the frame as raw bytes — no PNG, no disk on
// this path — so perception sees each moment with the lowest possible latency. SECONDARY
// path = the JSONL + PNG recorder, kept so sessions can still be replayed to TRAIN and
// evaluate D1/D2. Both read the same per-tick packet; offline replay re-emits the same
// stream, so D1/D2 are written once against "a packet source", live or replayed.
//
// Each packet carries keys/mouse, crosshair, held item, hotbar, menu, look, position,
// POV frame, and the player's own block place/break events. Block events come from
// Fabric's player events on the server side (no world-gen noise), buffered in a
// thread-safe queue and drained into each tick's packet.
//
// POV frames: the framebuffer is read on the render thread (one GPU->CPU copy); the
// downscale, the socket send, and the PNG write all run off it. Captured at 20 fps to
// match the B0 contract and downscaled to a set width keeping the real aspect ratio (no
// stretch). The size floor comes from what the B1 models eat: MineCLIP takes a 256x160
// picture and VPT a 128x128 one, so a saved frame must never be smaller than 256 wide or
// 160 tall — shrinking below that would force upscaling later, which invents pixels.
// (B2 reads no frames at all; it works from block events.) Frames are skipped while a
// menu is open or the status overlay is up. VM options: -Dmica.frameEvery=N (default 1 =
// 20 fps), -Dmica.frameSize=PX (default 320 = frame width, floor 256; height follows the
// window's aspect with a 160 floor), -Dmica.livePort=N (default 25567).
//
// The status overlay (F8) shows live capture health and is OFF by default; frame capture
// pauses while it is on, so it can never reach the saved/streamed frames. Recording
// pauses/resumes with F9 (pauses the live feed too); paused ticks write nothing and hold
// the counter, so the recording stays gap-free.
public class B0CaptureClient implements ClientModInitializer {
    public static final Logger LOGGER = LogManager.getLogger("mica_b0");

    // The smallest frame the B1 pixel models can consume without upscaling:
    // MineCLIP needs 256x160, VPT needs 128x128. Anything at or above 256 wide
    // and 160 tall can be shrunk to either model's input without inventing pixels.
    private static final int MIN_FRAME_WIDTH = 256;
    private static final int MIN_FRAME_HEIGHT = 160;

    private volatile BufferedWriterState state;   // read on the server thread (block events) too
    private volatile LiveStream live;             // read on the frame-writer thread too
    private KeyMapping toggleHud;
    private KeyMapping toggleRecording;
    private boolean hudVisible = false;
    private volatile boolean recording = true;   // F9 pauses/resumes; starts on so nothing is lost

    @Override
    public void onInitializeClient() {
        LOGGER.info("[MICA] B0 capture mod loaded.");
        state = openSession();
        live = openLiveStream();
        toggleHud = KeyBindingHelper.registerKeyBinding(new KeyMapping(
                "key.mica.toggle_hud", InputConstants.Type.KEYSYM, GLFW.GLFW_KEY_F8, "category.mica"));
        toggleRecording = KeyBindingHelper.registerKeyBinding(new KeyMapping(
                "key.mica.toggle_recording", InputConstants.Type.KEYSYM, GLFW.GLFW_KEY_F9, "category.mica"));
        ClientTickEvents.END_CLIENT_TICK.register(this::onClientTick);
        ClientLifecycleEvents.CLIENT_STOPPING.register(this::onClientStopping);
        HudRenderCallback.EVENT.register((poseStack, tickDelta) -> drawHud(poseStack));
        registerBlockEvents();
    }

    // All per-session state lives together so a failed open just disables capture.
    private static final class BufferedWriterState {
        java.io.BufferedWriter writer;
        Path jsonlPath;
        Path manifestPath;
        Path framesDir;
        String sessionId;
        long sessionStartMs;
        int tick = 0;
        int frameEvery;
        int frameSize;
        ExecutorService frameWriter;
        long readbackTotalNs = 0;
        long lastReadbackNs = 0;
        int frameCount = 0;
        final Queue<JsonObject> eventQueue = new ConcurrentLinkedQueue<>();
        final AtomicInteger eventCounter = new AtomicInteger(0);
    }

    private BufferedWriterState openSession() {
        try {
            BufferedWriterState st = new BufferedWriterState();
            st.sessionId = "fabric-" + LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss"));
            String dirProp = System.getProperty("mica.captureDir", "D:/2026projects/MICA/capture/raw");
            Path dir = Paths.get(dirProp);
            Files.createDirectories(dir);
            st.jsonlPath = dir.resolve(st.sessionId + ".jsonl");
            st.manifestPath = dir.resolve(st.sessionId + ".manifest.json");
            st.framesDir = dir.resolve(st.sessionId).resolve("frames");
            Files.createDirectories(st.framesDir);
            st.writer = Files.newBufferedWriter(st.jsonlPath, StandardCharsets.UTF_8,
                    StandardOpenOption.CREATE, StandardOpenOption.APPEND);
            st.frameEvery = Math.max(1, Integer.getInteger("mica.frameEvery", 1));   // 1 = 20 fps (contract)
            st.frameSize = Math.max(MIN_FRAME_WIDTH, Integer.getInteger("mica.frameSize", 320));  // frame width; floored so B1's models never upscale
            st.frameWriter = Executors.newSingleThreadExecutor(r -> {
                Thread th = new Thread(r, "mica-frame-writer");
                th.setDaemon(true);
                return th;
            });
            st.sessionStartMs = System.currentTimeMillis();
            writeManifest(st, -1);   // provisional: a crash now still leaves an ingestable manifest
            LOGGER.info("[MICA] recording to " + st.jsonlPath
                    + " (frames every " + st.frameEvery + " ticks, width " + st.frameSize + "px)");
            return st;
        } catch (IOException e) {
            LOGGER.error("[MICA] could not open capture session", e);
            return null;
        }
    }

    // The player's own block changes, on the server side (no world-gen noise).
    private void registerBlockEvents() {
        PlayerBlockBreakEvents.AFTER.register((world, player, pos, blockState, blockEntity) -> {
            if (world.isClientSide) return;
            String blockType = Registry.BLOCK.getKey(blockState.getBlock()).toString();
            recordEvent(pos.getX(), pos.getY(), pos.getZ(), blockType, "break", player.getName().getString());
        });
        UseBlockCallback.EVENT.register((player, world, hand, hit) -> {
            if (!world.isClientSide) {
                ItemStack held = player.getItemInHand(hand);
                if (held.getItem() instanceof BlockItem) {
                    // A block lands on the clicked cell when that cell is replaceable
                    // (water, grass, snow...), otherwise on the cell against the clicked
                    // face. Testing "replaceable" (not just "air") keeps placements into
                    // water/grass/snow from being missed.
                    BlockPos clicked = hit.getBlockPos();
                    BlockPos placePos = world.getBlockState(clicked).getMaterial().isReplaceable()
                            ? clicked
                            : clicked.relative(hit.getDirection());
                    if (world.getBlockState(placePos).getMaterial().isReplaceable()) {
                        String blockType = Registry.BLOCK.getKey(((BlockItem) held.getItem()).getBlock()).toString();
                        recordEvent(placePos.getX(), placePos.getY(), placePos.getZ(), blockType, "place",
                                player.getName().getString());
                    }
                }
            }
            return InteractionResult.PASS;   // observe only, don't change behaviour
        });
    }

    // Called from the server thread; the per-tick drain (client thread) reads the queue.
    private void recordEvent(int x, int y, int z, String blockType, String op, String actor) {
        if (state == null || !recording) return;   // drop block changes made while paused
        JsonObject ev = new JsonObject();
        ev.addProperty("event_id", state.eventCounter.getAndIncrement());
        JsonArray pos = new JsonArray();
        pos.add(x);
        pos.add(y);
        pos.add(z);
        ev.add("pos", pos);
        ev.addProperty("block_type", blockType);
        ev.addProperty("op", op);
        ev.addProperty("actor", actor);
        state.eventQueue.add(ev);
    }

    private void onClientTick(Minecraft client) {
        if (state == null) return;
        while (toggleHud.consumeClick()) hudVisible = !hudVisible;        // show/hide the overlay
        boolean wasRecording = recording;
        while (toggleRecording.consumeClick()) recording = !recording;    // pause/resume capture
        boolean justPaused = wasRecording && !recording;                  // record one last tick to flush
        LocalPlayer player = client.player;
        if (player == null || client.level == null) return;  // only record while in a world
        if (!recording && !justPaused) return;               // paused: write nothing, hold the counter
        final int t = state.tick++;
        long now = System.currentTimeMillis();

        JsonObject packet = new JsonObject();
        packet.addProperty("tick", t);
        packet.addProperty("wallclock_ms", now);

        JsonObject c = new JsonObject();
        c.addProperty("capture_wallclock_ms", now);
        c.add("pov_frame", capturePovFrame(client, t));
        c.add("input_state", inputState(client));
        c.addProperty("yaw", player.yRot);
        c.addProperty("pitch", player.xRot);
        c.add("crosshair_target", crosshair(client));
        c.addProperty("held_item", itemId(player.getMainHandItem()));
        c.add("hotbar", hotbar(player));
        c.addProperty("gui_open", client.screen != null);
        packet.add("client", c);

        JsonObject s = new JsonObject();
        JsonArray pos = new JsonArray();
        pos.add(player.getX());
        pos.add(player.getY());
        pos.add(player.getZ());
        s.add("player_pos", pos);
        s.add("block_events", drainEvents());
        s.add("inventory_delta", new JsonArray());
        s.addProperty("dimension", client.level.dimension().location().toString());
        s.addProperty("biome", "unknown");
        packet.add("server", s);

        String line = packet.toString();
        if (live != null) live.offer(packetMessage(line));   // live stream (primary): symbolic out now
        try {
            state.writer.write(line);                        // recorder (secondary)
            state.writer.write("\n");
            if (t % 20 == 0) state.writer.flush();            // cap data loss to ~1s if it crashes
        } catch (IOException e) {
            LOGGER.error("[MICA] write failed", e);
        }
    }

    // Move every block change seen since the last tick into this tick's packet.
    private JsonArray drainEvents() {
        JsonArray events = new JsonArray();
        JsonObject ev;
        while ((ev = state.eventQueue.poll()) != null) {
            events.add(ev);
        }
        return events;
    }

    // Read the framebuffer on the render thread (cheap), hand the rest off-thread.
    // Throttled, and skipped while a menu is open or the overlay is up. Returns the
    // pov_frame JSON, or null on a non-capture tick / failure.
    private JsonElement capturePovFrame(Minecraft client, int t) {
        if (state.frameWriter == null || t % state.frameEvery != 0 || client.screen != null || hudVisible) {
            return JsonNull.INSTANCE;
        }
        try {
            RenderTarget fb = client.getMainRenderTarget();
            int fbW = fb.viewWidth;
            int fbH = fb.viewHeight;
            long t0 = System.nanoTime();
            NativeImage img = new NativeImage(fbW, fbH, false);
            RenderSystem.bindTexture(fb.getColorTextureId());
            img.downloadTexture(0, true);
            img.flipY();
            long readbackNs = System.nanoTime() - t0;
            state.readbackTotalNs += readbackNs;
            state.lastReadbackNs = readbackNs;
            state.frameCount++;
            // Downscale to the set width, keeping the real aspect ratio so the saved/
            // streamed frame isn't stretched (VPT/MineCLIP expect undistorted views).
            // A very wide window would make the frame shorter than MineCLIP's input;
            // in that case grow the whole frame (still aspect-true) instead.
            // Assigned exactly once so the lambda below can capture them.
            int scaledH = Math.max(1, Math.round((float) state.frameSize * fbH / fbW));
            boolean tooShort = scaledH < MIN_FRAME_HEIGHT;
            int outW = tooShort ? Math.round((float) MIN_FRAME_HEIGHT * fbW / fbH) : state.frameSize;
            int outH = tooShort ? MIN_FRAME_HEIGHT : scaledH;
            Path out = state.framesDir.resolve(t + ".png");
            state.frameWriter.submit(() -> writeFrame(img, out, outW, outH, t));
            JsonObject f = new JsonObject();
            f.addProperty("path", out.toString());
            f.addProperty("width", outW);
            f.addProperty("height", outH);
            return f;
        } catch (Throwable e) {
            LOGGER.error("[MICA] frame capture failed", e);
            return JsonNull.INSTANCE;
        }
    }

    // Runs on the writer thread: downscale once, then stream the raw bytes (primary) and
    // save the PNG (recorder). Both come off the one downscale; neither blocks the game.
    private void writeFrame(NativeImage img, Path out, int outW, int outH, int tick) {
        NativeImage small = null;
        try {
            small = new NativeImage(outW, outH, false);
            img.resizeSubRectTo(0, 0, img.getWidth(), img.getHeight(), small);
            if (live != null) live.offer(frameMessage(tick, small));   // live stream (primary): raw RGBA
            small.writeToFile(out.toFile());                           // recorder (secondary): PNG to disk
        } catch (Throwable e) {
            LOGGER.error("[MICA] frame write failed", e);
        } finally {
            if (small != null) small.close();
            img.close();
        }
    }

    // On-screen capture-health readout. OFF by default; while it is on, frame capture is
    // paused (see capturePovFrame) so it can never contaminate a saved/streamed frame.
    //
    // Colour code, kept consistent so it's never ambiguous:
    //   green  = healthy / recording / frames flowing
    //   red    = NOT recording (capture failed to open) — the only thing red ever means
    //   yellow = paused or warming up (a menu is open, or no frame written yet) — expected
    //   grey   = plain numbers
    private void drawHud(PoseStack poseStack) {
        if (!hudVisible) return;
        Minecraft client = Minecraft.getInstance();
        if (client.level == null) return;               // nothing to show outside a world
        Font font = client.font;
        int x = 4;
        int y = 4;
        int step = font.lineHeight + 2;

        if (state == null) {                            // session never opened -> truly not recording
            font.drawShadow(poseStack, "MICA B0  NOT RECORDING", x, y, 0xFF5555);
            return;
        }

        if (recording) {
            font.drawShadow(poseStack, "MICA B0  RECORDING", x, y, 0x55FF55);   // green = writing
        } else {
            font.drawShadow(poseStack, "MICA B0  PAUSED (F9)", x, y, 0xFFFF55);  // yellow = paused
        }
        y += step;
        font.drawShadow(poseStack, "tick " + state.tick, x, y, 0xFFFFFF);
        y += step;
        int framesColor = state.frameCount > 0 ? 0x55FF55 : 0xFFFF55;   // yellow until the first frame
        font.drawShadow(poseStack, "frames " + state.frameCount + " (every " + state.frameEvery + ")", x, y, framesColor);
        y += step;
        font.drawShadow(poseStack, "block events " + state.eventCounter.get(), x, y, 0xFFFFFF);
        y += step;
        int liveColor = (live != null && live.hasConsumer()) ? 0x55FF55 : 0xAAAAAA;
        font.drawShadow(poseStack, "live " + (live == null ? "off" : live.hasConsumer() ? "connected" : "waiting"), x, y, liveColor);
        y += step;
        font.drawShadow(poseStack, "readback " + (state.lastReadbackNs / 1000) + " us", x, y, 0xAAAAAA);
        y += step;
        font.drawShadow(poseStack, "frames pause while this HUD is up", x, y, 0xFFFF55);
    }

    private JsonObject inputState(Minecraft client) {
        JsonObject in = new JsonObject();
        JsonArray keys = new JsonArray();
        if (client.options.keyUp.isDown()) keys.add("forward");
        if (client.options.keyDown.isDown()) keys.add("back");
        if (client.options.keyLeft.isDown()) keys.add("left");
        if (client.options.keyRight.isDown()) keys.add("right");
        if (client.options.keyJump.isDown()) keys.add("jump");
        if (client.options.keyShift.isDown()) keys.add("sneak");
        if (client.options.keySprint.isDown()) keys.add("sprint");
        in.add("keys", keys);
        JsonArray mouse = new JsonArray();
        if (client.options.keyAttack.isDown()) mouse.add("left");
        if (client.options.keyUse.isDown()) mouse.add("right");
        in.add("mouse_buttons", mouse);
        in.addProperty("mouse_dx", 0.0);   // raw mouse delta not captured; look change is in yaw/pitch
        in.addProperty("mouse_dy", 0.0);
        return in;
    }

    private JsonElement crosshair(Minecraft client) {
        HitResult hit = client.hitResult;
        if (hit == null || hit.getType() == HitResult.Type.MISS) return JsonNull.INSTANCE;
        JsonObject ch = new JsonObject();
        if (hit.getType() == HitResult.Type.BLOCK) {
            BlockHitResult bhr = (BlockHitResult) hit;
            BlockPos p = bhr.getBlockPos();
            JsonArray pos = new JsonArray();
            pos.add(p.getX());
            pos.add(p.getY());
            pos.add(p.getZ());
            ch.add("block_pos", pos);
            ch.addProperty("face", bhr.getDirection().getName());
            ch.add("entity", JsonNull.INSTANCE);
        } else {
            ch.add("block_pos", JsonNull.INSTANCE);
            ch.add("face", JsonNull.INSTANCE);
            EntityHitResult ehr = (EntityHitResult) hit;
            ch.addProperty("entity", ehr.getEntity().getName().getString());
        }
        return ch;
    }

    private JsonArray hotbar(LocalPlayer player) {
        JsonArray hb = new JsonArray();
        for (int i = 0; i < 9; i++) hb.add(itemId(player.inventory.getItem(i)));
        return hb;
    }

    private String itemId(ItemStack stack) {
        if (stack == null || stack.isEmpty()) return "minecraft:air";
        return Registry.ITEM.getKey(stack.getItem()).toString();
    }

    // Writes the session manifest. Called twice: provisionally at open (declaredCount=-1,
    // so a crash still leaves an ingestable file), then with the real total at clean stop.
    private void writeManifest(BufferedWriterState st, int declaredCount) {
        JsonObject m = new JsonObject();
        m.addProperty("session_id", st.sessionId);
        m.addProperty("session_start_ms", st.sessionStartMs);
        m.addProperty("mc_version", "1.16.5");
        m.addProperty("mod_version", "fabric-b0-0.0.2");
        m.addProperty("event_schema_version", "1");   // B0 block-event schema version (jsonl_ingest reads this)
        // The frame settings this recording ran with — two captures with different
        // settings must be tellable apart from their manifests alone.
        m.addProperty("frame_every", st.frameEvery);
        m.addProperty("frame_width_px", st.frameSize);
        m.addProperty("declared_event_count", declaredCount);
        try {
            Files.write(st.manifestPath, m.toString().getBytes(StandardCharsets.UTF_8));
        } catch (IOException e) {
            LOGGER.error("[MICA] manifest write failed", e);
        }
    }

    // ---- live stream (primary feed to D1/D2) --------------------------------------

    private LiveStream openLiveStream() {
        int port = Math.max(1, Integer.getInteger("mica.livePort", 25567));
        try {
            LiveStream stream = new LiveStream(port);
            LOGGER.info("[MICA] live B0 stream on 127.0.0.1:" + port);
            return stream;
        } catch (IOException e) {
            LOGGER.error("[MICA] live stream disabled (could not bind port " + port + ")", e);
            return null;
        }
    }

    private static byte[] packetMessage(String json) {
        return framed((byte) 0, json.getBytes(StandardCharsets.UTF_8));
    }

    // [4-byte tick][raw RGBA bytes, row-major, top-left origin]. Read off-thread.
    private static byte[] frameMessage(int tick, NativeImage small) {
        int w = small.getWidth();
        int h = small.getHeight();
        byte[] payload = new byte[4 + w * h * 4];
        payload[0] = (byte) (tick >>> 24);
        payload[1] = (byte) (tick >>> 16);
        payload[2] = (byte) (tick >>> 8);
        payload[3] = (byte) tick;
        int i = 4;
        for (int yy = 0; yy < h; yy++) {
            for (int xx = 0; xx < w; xx++) {
                int px = small.getPixelRGBA(xx, yy);   // bytes in memory are R, G, B, A
                payload[i++] = (byte) px;
                payload[i++] = (byte) (px >> 8);
                payload[i++] = (byte) (px >> 16);
                payload[i++] = (byte) (px >> 24);
            }
        }
        return framed((byte) 1, payload);
    }

    private static byte[] framed(byte type, byte[] payload) {
        byte[] msg = new byte[5 + payload.length];
        msg[0] = type;
        int len = payload.length;
        msg[1] = (byte) (len >>> 24);
        msg[2] = (byte) (len >>> 16);
        msg[3] = (byte) (len >>> 8);
        msg[4] = (byte) len;
        System.arraycopy(payload, 0, msg, 5, payload.length);
        return msg;
    }

    // Owns the socket. One background thread accepts a consumer and streams the queue to
    // it, re-accepting if it drops. offer() is called from game threads and never blocks:
    // if the consumer falls behind, the OLDEST message is dropped so frames stay fresh and
    // the game never stalls. Lossy on purpose — the disk recorder is the complete copy.
    private static final class LiveStream {
        private final ServerSocket server;
        private final BlockingQueue<byte[]> queue = new ArrayBlockingQueue<>(64);
        private final Thread thread;
        private volatile Socket client;

        LiveStream(int port) throws IOException {
            server = new ServerSocket();
            server.setReuseAddress(true);
            server.bind(new InetSocketAddress("127.0.0.1", port));
            thread = new Thread(this::run, "mica-live-stream");
            thread.setDaemon(true);
            thread.start();
        }

        private void run() {
            while (!server.isClosed()) {
                try (Socket sock = server.accept()) {
                    sock.setTcpNoDelay(true);
                    queue.clear();           // start the new consumer fresh, no stale backlog
                    client = sock;
                    OutputStream out = sock.getOutputStream();
                    while (true) {
                        out.write(queue.take());
                        out.flush();
                    }
                } catch (Exception e) {
                    client = null;           // consumer gone (or shutting down) — wait for the next
                }
            }
        }

        boolean hasConsumer() {
            return client != null;
        }

        void offer(byte[] msg) {
            if (client == null) return;      // no consumer: don't buffer
            if (!queue.offer(msg)) {
                queue.poll();                // full: drop the oldest, keep the newest
                queue.offer(msg);
            }
        }

        void close() {
            thread.interrupt();   // unblock queue.take()/accept() so the stream thread exits promptly
            try {
                server.close();
            } catch (IOException ignored) {
            }
            Socket c = client;
            if (c != null) {
                try {
                    c.close();
                } catch (IOException ignored) {
                }
            }
        }
    }

    private void onClientStopping(Minecraft client) {
        if (live != null) {
            live.close();
            live = null;
        }
        if (state == null) return;
        if (state.frameWriter != null) {
            state.frameWriter.shutdown();
            try {
                state.frameWriter.awaitTermination(10, TimeUnit.SECONDS);
            } catch (InterruptedException ignored) {
            }
        }
        try {
            state.writer.flush();
            state.writer.close();
        } catch (IOException ignored) {
        }
        writeManifest(state, state.eventCounter.get());   // final count overwrites the provisional one
        if (state.frameCount > 0) {
            LOGGER.info("[MICA] avg frame read-back: " + (state.readbackTotalNs / state.frameCount / 1000)
                    + " us over " + state.frameCount + " frames");
        }
        LOGGER.info("[MICA] wrote " + state.tick + " moments, " + state.eventCounter.get()
                + " block events, to " + state.jsonlPath);
        state = null;
    }
}
