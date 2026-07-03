package mica.capture.mixin;

import mica.capture.B0CaptureClient;
import net.minecraft.world.InteractionResult;
import net.minecraft.world.item.BlockItem;
import net.minecraft.world.item.context.BlockPlaceContext;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

// The game's own "a block was really placed" moment. BlockItem.place is where vanilla
// decides success or failure for every ordinary block item, and by this point
// BlockPlaceContext has already resolved the real target cell (the clicked cell when it
// was replaceable — water, grass, snow — otherwise the neighbor against the clicked
// face). Recording here, only on success, replaces the old click-time prediction, which
// logged phantom placements (a click on a chest with planks in hand, a rejected
// build-height click) and could never see a rejection.
//
// Known gap, on purpose: an item that places a second block elsewhere (door tops, bed
// halves, tall plants) records only the clicked cell. D2's replay-vs-snapshot check
// surfaces those as divergence, and the v1 build templates avoid such blocks.
@Mixin(BlockItem.class)
public abstract class BlockItemMixin {

    @Inject(method = "place", at = @At("RETURN"))
    private void mica$recordPlacement(BlockPlaceContext context, CallbackInfoReturnable<InteractionResult> cir) {
        if (cir.getReturnValue().consumesAction()) {
            B0CaptureClient.recordAuthoritativePlace(context.getLevel(), context.getClickedPos(), context.getPlayer());
        }
    }
}
