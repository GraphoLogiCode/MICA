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
import net.minecraft.client.KeyMapping;
import net.minecraft.client.Minecraft;
import net.minecraft.client.gui.Font;
import net.minecraft.client.player.LocalPlayer;
import net.minecraft.core.BlockPos;
import net.minecraft.core.Registry;
import net.minecraft.world.entity.player.Player;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.level.Level;
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
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
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
// POV frame, and the player's own block place/break events. Break events come from
// Fabric's server-side player event; place events come from a mixin inside
// BlockItem.place, AFTER the game accepted the placement — so what is recorded is what
// actually landed, not what a click predicted (see BlockItemMixin for the known
// door/bed gap). Both are buffered in a thread-safe queue and drained into each
// tick's packet.
//
// REGION SNAPSHOTS (for D2): the ground truth its event-replay is checked against.
// A fixed box around the player's starting spot is saved as palette + run-length
// counts — once at the start, then again after each building burst, at a quiet
// moment (no block change for 2 s) because the client's copy of the world can trail
// the server by a tick right after a change. Snapshots store block identity only
// (no facing/waterlogged state), matching what the event stream carries. Files:
// <session>/snapshots/<tick>.json; the box is recorded in the manifest.
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
// pauses while it is on, so it can never reach the saved/streamed frames.
//
// There is deliberately NO pause key (0.0.5). The old F9 pause dropped block events for
// changes made while paused and held the tick counter, so the recording LOOKED gap-free
// while the world silently moved on without evidence — one accidental press (F9 sat next
// to F8, with no banner unless the HUD was open) cost a real session its structure
// evidence, permanently. Recording runs whenever a world is open; to stop capturing,
// quit the world — that is a clean session end.
public class B0CaptureClient implements ClientModInitializer {
    public static final Logger LOGGER = LogManager.getLogger("mica_b0");

    // The smallest frame the B1 pixel models can consume without upscaling:
    // MineCLIP needs 256x160, VPT needs 128x128. Anything at or above 256 wide
    // and 160 tall can be shrunk to either model's input without inventing pixels.
    private static final int MIN_FRAME_WIDTH = 256;
    private static final int MIN_FRAME_HEIGHT = 160;

    // Region snapshots wait for this many event-free ticks (2 s) so the client's copy
    // of the world has settled — right after a change it can trail the server by a tick.
    private static final int QUIESCENT_TICKS = 40;
    // The snapshot box around the player. Builds are capped to roughly 16x16x16
    // (the D2 templates), so radius 24 leaves room to wander while placing.
    private static final int REGION_RADIUS_DEFAULT = 24;   // horizontal, blocks
    private static final int REGION_BELOW = 16;            // how far under the feet to include
    private static final int REGION_ABOVE = 32;            // headroom for towers
    // Until the first block change, the region FOLLOWS the player: it re-centers when
    // they move more than this far from the current anchor (a real session anchored at
    // the spawn while the player teleported away to build — every event then fell
    // outside the box and the structure stream was blind for the whole session).
    private static final int REANCHOR_BLOCKS = 8;
    // Region v3 (0.0.6): after the freeze the box GROWS toward the player. Two real
    // sessions were voided by a stray first block pinning the box away from the
    // build; growth (never moving, never shrinking) admits new territory with its
    // base snapshotted BEFORE the player can build there. -Dmica.regionGrow=0 opts out.
    private static final int GROW_MARGIN = 8;          // grow when the player is this close to a face
    private static final int GROW_STEP = 16;           // how far past the player each face extends
    private static final int GROW_COOLDOWN_TICKS = 40; // at most one growth per 2 s
    private static final int GROW_QUIET_TICKS = 5;     // short quiet so the compare misses the trailing tick
    private static final int GROW_MAX_DIMENSION = 145; // past this, escapes resume being the signal

    // The mixin (server thread) needs a path to the running mod instance.
    private static volatile B0CaptureClient INSTANCE;

    private volatile BufferedWriterState state;   // read on the server thread (block events) too
    private volatile LiveStream live;             // read on the frame-writer thread too
    private KeyMapping toggleHud;
    private boolean hudVisible = false;

    @Override
    public void onInitializeClient() {
        LOGGER.info("[MICA] B0 capture mod loaded.");
        INSTANCE = this;
        state = openSession();
        live = openLiveStream();
        toggleHud = KeyBindingHelper.registerKeyBinding(new KeyMapping(
                "key.mica.toggle_hud", InputConstants.Type.KEYSYM, GLFW.GLFW_KEY_F8, "category.mica"));
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
        // Counts what THIS recorder records — a change the hooks never see is invisible
        // to it. The manifest's declared_event_count therefore proves the recorder agrees
        // with itself, not that nothing in the world was missed (D0 2026-07-02 amendment).
        // With the authoritative place mixin the hooks see every ordinary placement, and
        // the final arbiter of "nothing missed" is D2's replay-vs-snapshot check.
        final AtomicInteger eventCounter = new AtomicInteger(0);
        // Region snapshots (D2 ground truth). The box first anchors wherever the player
        // is, FOLLOWS them while nothing has been built (re-anchoring as they move),
        // and freezes for good at the first block change — so it frames the BUILD,
        // not the spawn point.
        Path snapshotsDir;
        int[] snapshotRegion;               // {x0, y0, z0, x1, y1, z1}, inclusive
        boolean regionFrozen = false;       // set at the first recorded block change
        int lastBaseTick = -1;              // tick of the current provisional base snapshot
        int lastGrowTick = 0;               // tick of the last grow-only expansion (region v3)
        int quietTicks = 0;                 // ticks since the last recorded block event
        int eventsSinceSnapshot = 0;
        int snapshotCount = 0;
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
            st.snapshotsDir = dir.resolve(st.sessionId).resolve("snapshots");
            Files.createDirectories(st.framesDir);
            Files.createDirectories(st.snapshotsDir);
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

    // The player's own block breaks, on the server side (no world-gen noise). Places are
    // recorded by BlockItemMixin — inside the game's own success path, not from clicks.
    private void registerBlockEvents() {
        PlayerBlockBreakEvents.AFTER.register((world, player, pos, blockState, blockEntity) -> {
            if (world.isClientSide) return;
            String blockType = Registry.BLOCK.getKey(blockState.getBlock()).toString();
            recordEvent(pos.getX(), pos.getY(), pos.getZ(), blockType, "break", player.getName().getString());
        });
    }

    // Called by BlockItemMixin (server thread) after the game ACCEPTED a placement. The
    // world state at pos is read back, so the event carries what actually landed — a
    // stair keeps its block id, a rejected click never reaches here. player is null for
    // machine placements (dispensers); those are not the human's hand and are skipped.
    public static void recordAuthoritativePlace(Level world, BlockPos pos, Player player) {
        B0CaptureClient mod = INSTANCE;
        if (mod == null || world.isClientSide || player == null) return;
        String blockType = Registry.BLOCK.getKey(world.getBlockState(pos).getBlock()).toString();
        mod.recordEvent(pos.getX(), pos.getY(), pos.getZ(), blockType, "place", player.getName().getString());
    }

    // Called from the server thread; the per-tick drain (client thread) reads the queue.
    private void recordEvent(int x, int y, int z, String blockType, String op, String actor) {
        if (state == null) return;
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
        LocalPlayer player = client.player;
        if (player == null || client.level == null) return;  // only record while in a world
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
        JsonArray drained = drainEvents();
        s.add("block_events", drained);
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
        maybeSnapshot(client, player, t, drained.size());
    }

    // ---- region snapshots (D2 ground truth) ----------------------------------------

    // Take the first snapshot as soon as we know where the player is, then one after
    // each building burst — at a quiet moment, so the client's world copy has settled
    // and every event already drained into a packet belongs strictly before it.
    //
    // Until the first block change the region is PROVISIONAL: it follows the player
    // (re-centering when they move more than REANCHOR_BLOCKS from the anchor, at most
    // once per QUIESCENT_TICKS), and each re-anchor rewrites the base snapshot — which
    // stays strictly pre-action by construction, because no event exists yet. The
    // first recorded block change freezes the region where the builder actually is.
    private void maybeSnapshot(Minecraft client, LocalPlayer player, int tick, int eventsThisTick) {
        if (state.snapshotRegion == null) {
            initSnapshotRegion(player);
            writeSnapshot(client, tick);
            state.lastBaseTick = tick;
            return;
        }
        if (!state.regionFrozen && state.eventCounter.get() > 0) {
            state.regionFrozen = true;
            pruneProvisionalSnapshots();   // only the final region's base remains
            LOGGER.info("[MICA] snapshot region FROZEN at the first build: "
                    + java.util.Arrays.toString(state.snapshotRegion));
        }
        if (!state.regionFrozen) {
            int px = (int) Math.floor(player.getX());
            int pz = (int) Math.floor(player.getZ());
            int cx = (state.snapshotRegion[0] + state.snapshotRegion[3]) / 2;
            int cz = (state.snapshotRegion[2] + state.snapshotRegion[5]) / 2;
            long dx = px - cx;
            long dz = pz - cz;
            boolean moved = dx * dx + dz * dz > (long) REANCHOR_BLOCKS * REANCHOR_BLOCKS;
            if (moved && tick - state.lastBaseTick >= QUIESCENT_TICKS) {
                int staleBase = state.lastBaseTick;
                initSnapshotRegion(player);            // re-center + persist to manifest
                deleteSnapshotFile(staleBase);         // keep one provisional base at a time
                writeSnapshot(client, tick);
                state.lastBaseTick = tick;
            }
            return;   // no burst snapshots while provisional — nothing was built yet
        }
        maybeGrowRegion(client, player, tick, eventsThisTick);
        if (eventsThisTick > 0) {
            state.quietTicks = 0;
            state.eventsSinceSnapshot += eventsThisTick;
            return;
        }
        state.quietTicks++;
        if (state.eventsSinceSnapshot > 0 && state.quietTicks == QUIESCENT_TICKS) {
            writeSnapshot(client, tick);
        }
    }

    // Region v3 (0.0.6): the frozen box follows the builder by GROWING — never moving,
    // never shrinking, so no existing cell's base or coordinate frame ever changes.
    // When the player comes within GROW_MARGIN of a face (or stands beyond it), the
    // crossed faces extend GROW_STEP past them, the manifest is rewritten (it always
    // carries the CURRENT box), and the enlarged region is snapshotted IMMEDIATELY:
    // that file is both a comparison point for the old cells and the pre-build BASE
    // for the adopted ones — captured before the player can build there, which is
    // exactly what the margin buys. A short quiet requirement keeps the comparison
    // off the trailing tick right after a change; the size cap returns escapes as
    // the honest signal for a build no box should chase.
    private void maybeGrowRegion(Minecraft client, LocalPlayer player, int tick, int eventsThisTick) {
        if (!"1".equals(System.getProperty("mica.regionGrow", "1"))) return;
        if (eventsThisTick > 0 || state.quietTicks < GROW_QUIET_TICKS) return;
        if (tick - state.lastGrowTick < GROW_COOLDOWN_TICKS) return;
        int[] r = state.snapshotRegion;
        int px = (int) Math.floor(player.getX());
        int py = (int) Math.floor(player.getY());
        int pz = (int) Math.floor(player.getZ());
        int[] g = java.util.Arrays.copyOf(r, 6);
        if (px - r[0] < GROW_MARGIN) g[0] = Math.min(g[0], px - GROW_STEP);
        if (r[3] - px < GROW_MARGIN) g[3] = Math.max(g[3], px + GROW_STEP);
        if (pz - r[2] < GROW_MARGIN) g[2] = Math.min(g[2], pz - GROW_STEP);
        if (r[5] - pz < GROW_MARGIN) g[5] = Math.max(g[5], pz + GROW_STEP);
        if (py - r[1] < GROW_MARGIN) g[1] = Math.max(0, Math.min(g[1], py - GROW_STEP));
        if (r[4] - py < GROW_MARGIN) g[4] = Math.min(255, Math.max(g[4], py + GROW_STEP));
        // per-axis cap: a capped axis keeps its old faces and escapes stay honest there
        if (g[3] - g[0] + 1 > GROW_MAX_DIMENSION) { g[0] = r[0]; g[3] = r[3]; }
        if (g[5] - g[2] + 1 > GROW_MAX_DIMENSION) { g[2] = r[2]; g[5] = r[5]; }
        if (g[4] - g[1] + 1 > GROW_MAX_DIMENSION) { g[1] = r[1]; g[4] = r[4]; }
        if (java.util.Arrays.equals(g, r)) return;
        state.snapshotRegion = g;
        state.lastGrowTick = tick;
        writeManifest(state, -1);          // the manifest always carries the CURRENT box
        writeSnapshot(client, tick);       // the adopted cells' base, captured pre-build
        LOGGER.info("[MICA] region grew: " + java.util.Arrays.toString(r)
                + " -> " + java.util.Arrays.toString(g));
    }

    // On freeze: drop every snapshot file except the current base, so snapshots/ only
    // ever holds frames of the FINAL region (the replay check reads the folder blind).
    private void pruneProvisionalSnapshots() {
        try {
            java.io.File[] files = state.snapshotsDir.toFile().listFiles();
            if (files == null) return;
            String keep = state.lastBaseTick + ".json";
            for (java.io.File file : files) {
                if (file.getName().endsWith(".json") && !file.getName().equals(keep)) {
                    final Path stale = file.toPath();
                    state.frameWriter.submit(() -> {   // after any queued write finishes
                        try {
                            Files.deleteIfExists(stale);
                        } catch (IOException e) {
                            LOGGER.error("[MICA] stale snapshot delete failed", e);
                        }
                    });
                }
            }
        } catch (Throwable e) {
            LOGGER.error("[MICA] snapshot prune failed", e);
        }
    }

    private void deleteSnapshotFile(int tick) {
        if (tick < 0) return;
        final Path stale = state.snapshotsDir.resolve(tick + ".json");
        state.frameWriter.submit(() -> {
            try {
                Files.deleteIfExists(stale);
            } catch (IOException e) {
                LOGGER.error("[MICA] stale snapshot delete failed", e);
            }
        });
    }

    private void initSnapshotRegion(LocalPlayer player) {
        int radius = Math.max(8, Integer.getInteger("mica.snapshotRadius", REGION_RADIUS_DEFAULT));
        int px = (int) Math.floor(player.getX());
        int py = (int) Math.floor(player.getY());
        int pz = (int) Math.floor(player.getZ());
        state.snapshotRegion = new int[]{
                px - radius, Math.max(0, py - REGION_BELOW), pz - radius,
                px + radius, Math.min(255, py + REGION_ABOVE), pz + radius};
        writeManifest(state, -1);   // the box defines every snapshot's frame; persist it now, crash or not
        LOGGER.info("[MICA] snapshot region anchored (provisional until first build): "
                + java.util.Arrays.toString(state.snapshotRegion));
    }

    // Read the whole box on the client thread (a ~100k-cell read costs ~10 ms — felt as
    // a tiny hitch, which is why it only runs at quiet moments), squeeze it down to a
    // palette plus run-lengths (uniform ground and air collapse to a handful of runs),
    // and hand the file write to the frame-writer thread.
    private void writeSnapshot(Minecraft client, int tick) {
        long t0 = System.nanoTime();
        int[] r = state.snapshotRegion;
        List<String> palette = new ArrayList<>();
        Map<String, Integer> indexOf = new HashMap<>();
        JsonArray runs = new JsonArray();
        int runIndex = -1;
        int runLength = 0;
        BlockPos.MutableBlockPos cursor = new BlockPos.MutableBlockPos();
        // Scan order y -> z -> x; the Python side must walk the same order to decode.
        for (int y = r[1]; y <= r[4]; y++) {
            for (int z = r[2]; z <= r[5]; z++) {
                for (int x = r[0]; x <= r[3]; x++) {
                    cursor.set(x, y, z);
                    String id = Registry.BLOCK.getKey(client.level.getBlockState(cursor).getBlock()).toString();
                    Integer idx = indexOf.get(id);
                    if (idx == null) {
                        idx = palette.size();
                        indexOf.put(id, idx);
                        palette.add(id);
                    }
                    if (idx == runIndex) {
                        runLength++;
                    } else {
                        if (runLength > 0) runs.add(runPair(runLength, runIndex));
                        runIndex = idx;
                        runLength = 1;
                    }
                }
            }
        }
        if (runLength > 0) runs.add(runPair(runLength, runIndex));

        JsonObject snap = new JsonObject();
        snap.addProperty("tick", tick);
        JsonArray region = new JsonArray();
        for (int v : r) region.add(v);
        snap.add("region", region);
        JsonArray names = new JsonArray();
        for (String name : palette) names.add(name);
        snap.add("palette", names);
        snap.add("runs", runs);

        String json = snap.toString();
        Path out = state.snapshotsDir.resolve(tick + ".json");
        state.frameWriter.submit(() -> {
            try {
                Files.write(out, json.getBytes(StandardCharsets.UTF_8));
            } catch (IOException e) {
                LOGGER.error("[MICA] snapshot write failed", e);
            }
        });
        state.eventsSinceSnapshot = 0;
        state.snapshotCount++;
        LOGGER.info("[MICA] region snapshot @" + tick + " (" + runs.size() + " runs, "
                + (System.nanoTime() - t0) / 1_000_000 + " ms)");
    }

    private static JsonArray runPair(int count, int paletteIndex) {
        JsonArray pair = new JsonArray();
        pair.add(count);
        pair.add(paletteIndex);
        return pair;
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
    //   yellow = warming up (a menu is open, or no frame written yet) — expected
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

        font.drawShadow(poseStack, "MICA B0  RECORDING", x, y, 0x55FF55);   // green = writing
        y += step;
        font.drawShadow(poseStack, "tick " + state.tick, x, y, 0xFFFFFF);
        y += step;
        int framesColor = state.frameCount > 0 ? 0x55FF55 : 0xFFFF55;   // yellow until the first frame
        font.drawShadow(poseStack, "frames " + state.frameCount + " (every " + state.frameEvery + ")", x, y, framesColor);
        y += step;
        font.drawShadow(poseStack, "block events " + state.eventCounter.get(), x, y, 0xFFFFFF);
        y += step;
        font.drawShadow(poseStack, "snapshots " + state.snapshotCount, x, y, 0xFFFFFF);
        y += step;
        if (state.snapshotRegion == null) {
            font.drawShadow(poseStack, "region waiting", x, y, 0xFFFF55);
        } else if (state.regionFrozen) {
            font.drawShadow(poseStack, "region fixed", x, y, 0x55FF55);
        } else {
            int cx = (state.snapshotRegion[0] + state.snapshotRegion[3]) / 2;
            int cz = (state.snapshotRegion[2] + state.snapshotRegion[5]) / 2;
            font.drawShadow(poseStack, "region follows you @(" + cx + "," + cz + ")", x, y, 0xFFFF55);
        }
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
        m.addProperty("mod_version", "fabric-b0-0.0.6");   // 0.0.6: grow-only region (v3) — the box follows the builder
        m.addProperty("event_schema_version", "1");   // B0 block-event schema version (jsonl_ingest reads this)
        // The frame settings this recording ran with — two captures with different
        // settings must be tellable apart from their manifests alone.
        m.addProperty("frame_every", st.frameEvery);
        m.addProperty("frame_width_px", st.frameSize);
        // Snapshot declaration: the quiet-tick rule says this build takes region
        // snapshots at all; the box (written once known) frames every snapshot file.
        m.addProperty("snapshot_quiet_ticks", QUIESCENT_TICKS);
        if (st.snapshotRegion != null) {
            JsonArray region = new JsonArray();
            for (int v : st.snapshotRegion) region.add(v);
            m.add("snapshot_region", region);
        }
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
