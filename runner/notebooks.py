"""Mapping from stage label (e.g. "01", "08b") to course notebook files."""

STAGE_NOTEBOOKS = {
    "01": [
        "course/Part1_TheNaiveLoop/3_kvCache/part1_kv_1_theCostOfForgetting.ipynb",
        "course/Part1_TheNaiveLoop/3_kvCache/part1_kv_3_CCwriteBothLoops_helper.ipynb",
    ],
    "02": [
        "course/Part1_TheNaiveLoop/3_kvCache/part1_kv_2_theCostOfRemembering.ipynb",
        "course/Part1_TheNaiveLoop/3_kvCache/part1_kv_3_CCwriteBothLoops_helper.ipynb",
    ],
    "03": [
        "course/Part1_TheNaiveLoop/1_arithmetic/part1_arith_1_readVsCompute.ipynb",
        "course/Part1_TheNaiveLoop/1_arithmetic/part1_arith_2_whatTheCacheCosts.ipynb",
        "course/Part1_TheNaiveLoop/2_roofline/part1_roof_1_measureYourCard.ipynb",
        "course/Part1_TheNaiveLoop/2_roofline/part1_roof_2_theKnee.ipynb",
    ],
    "04": [
        "course/Part2_Batching/1_padding/part2_pad_1_freeSequences.ipynb",
        "course/Part2_Batching/1_padding/part2_pad_2_theSlotThatWaited.ipynb",
        "course/Part2_Batching/1_padding/part2_pad_3_CCpickTheBatchSize_helper.ipynb",
    ],
    "05": [
        "course/Part2_Batching/2_continuous/part2_cont_1_iterationLevel.ipynb",
        "course/Part2_Batching/2_continuous/part2_cont_2_latencyUnderLoad.ipynb",
        "course/Part2_Batching/2_continuous/part2_cont_3_CCwriteTheScheduler_helper.ipynb",
    ],
    "06": [
        "course/Part3_PagedAttention/1_blocks/part3_blk_1_fragmentation.ipynb",
        "course/Part3_PagedAttention/1_blocks/part3_blk_2_thePageTable.ipynb",
        "course/Part3_PagedAttention/1_blocks/part3_blk_3_CCbuildTheAllocator_helper.ipynb",
    ],
    "07": [
        "course/Part3_PagedAttention/2_gather/part3_gat_1_attentionFromScratch.ipynb",
        "course/Part3_PagedAttention/2_gather/part3_gat_2_throughThePageTable.ipynb",
        "course/Part3_PagedAttention/2_gather/part3_gat_3_CCwriteTheOracle_helper.ipynb",
    ],
    "08": [
        "course/Part3_PagedAttention/3_kernels/part3_kern_1_memoryTransactions.ipynb",
        "course/Part3_PagedAttention/3_kernels/part3_kern_2_fillingTheMachine.ipynb",
        "course/Part3_PagedAttention/3_kernels/part3_kern_3_CCcoalesceTheGather_helper.ipynb",
    ],
    "08b": [
        "course/Part3_PagedAttention/3_kernels/part3_kern_1_memoryTransactions.ipynb",
        "course/Part3_PagedAttention/3_kernels/part3_kern_3_CCcoalesceTheGather_helper.ipynb",
    ],
    "08c": [
        "course/Part3_PagedAttention/3_kernels/part3_kern_2_fillingTheMachine.ipynb",
    ],
    "09": [
        "course/Part3_PagedAttention/4_sharing/part3_shr_1_copyOnWrite.ipynb",
        "course/Part3_PagedAttention/4_sharing/part3_shr_2_prefixCache.ipynb",
        "course/Part3_PagedAttention/4_sharing/part3_shr_3_CCbuildThePrefixCache_helper.ipynb",
    ],
    "10": [
        "course/Part4_TheScheduler/1_admission/part4_adm_1_runningOutMidFlight.ipynb",
        "course/Part4_TheScheduler/1_admission/part4_adm_2_swapVsRecompute.ipynb",
        "course/Part4_TheScheduler/1_admission/part4_adm_3_CCbuildTheScheduler_helper.ipynb",
    ],
    "11": [
        "course/Part4_TheScheduler/2_chunked/part4_chk_1_theStall.ipynb",
        "course/Part4_TheScheduler/2_chunked/part4_chk_2_CCmixTheBatch_helper.ipynb",
    ],
    "12": [
        "course/Part5_MakingItFast/1_cudaGraphs/part5_cg_1_thePythonTax.ipynb",
        "course/Part5_MakingItFast/1_cudaGraphs/part5_cg_2_shapeBuckets.ipynb",
        "course/Part5_MakingItFast/1_cudaGraphs/part5_cg_3_CCcaptureAndBucket_helper.ipynb",
    ],
    "13": [
        "course/Part5_MakingItFast/2_sampling/part5_smp_1_fourKnobs.ipynb",
        "course/Part5_MakingItFast/2_sampling/part5_smp_2_oneVectorizedPass.ipynb",
        "course/Part5_MakingItFast/2_sampling/part5_smp_3_CCsampleItRight_helper.ipynb",
    ],
    "14": [
        "course/Part5_MakingItFast/3_detokenize/part5_dtk_1_mojibake.ipynb",
        "course/Part5_MakingItFast/3_detokenize/part5_dtk_2_CCstreamItProperly_helper.ipynb",
    ],
    "15": [
        "course/Part6_TheServer/1_async/part6_asy_1_theLoopThatMustNotBlock.ipynb",
        "course/Part6_TheServer/1_async/part6_asy_2_CCbuildTheEngine_helper.ipynb",
    ],
    "16": [
        "course/Part6_TheServer/2_metrics/part6_met_1_theNumbersThatMatter.ipynb",
        "course/Part6_TheServer/2_metrics/part6_met_2_CCwhatIsBroken_helper.ipynb",
    ],
    "17": [
        "course/Part7_ModernVLLM/1_speculative/part7_spc_1_guessAndCheck.ipynb",
        "course/Part7_ModernVLLM/1_speculative/part7_spc_2_CCproveItIsUnbiased_helper.ipynb",
    ],
    "18": [
        "course/Part7_ModernVLLM/2_quantization/part7_qnt_1_halveTheBytes.ipynb",
        "course/Part7_ModernVLLM/2_quantization/part7_qnt_2_CCwhereDidTheSpeedGo_helper.ipynb",
    ],
    "18b": [
        "course/Part7_ModernVLLM/2_quantization/part7_qnt_1_halveTheBytes.ipynb",
    ],
    "19": [
        "course/Part7_ModernVLLM/3_guided/part7_gdd_1_makeItImpossible.ipynb",
        "course/Part7_ModernVLLM/3_guided/part7_gdd_2_CCmaskTheLogits_helper.ipynb",
    ],
    "20": [
        "course/Part7_ModernVLLM/4_tensorParallel/part7_tp_1_columnThenRow.ipynb",
        "course/Part7_ModernVLLM/4_tensorParallel/part7_tp_2_CCcountTheCollectives_helper.ipynb",
    ],
    "21": [
        "course/Part8_TheCapstone/1_engine/part8_eng_1_theFlatBatch.ipynb",
        "course/Part8_TheCapstone/1_engine/part8_eng_2_oneRule.ipynb",
    ],
    "22": [
        "course/Part8_TheCapstone/1_engine/part8_eng_3_CCbuildTheScheduler_helper.ipynb",
    ],
    "23": [
        "course/Part8_TheCapstone/2_graphs/part8_gr_1_launchesNotBytes.ipynb",
    ],
    "24": [
        "course/Part7_ModernVLLM/2_quantization/part7_qnt_1_halveTheBytes.ipynb",
    ],
    "24b": [
        "course/Part3_PagedAttention/3_kernels/part3_kern_1_memoryTransactions.ipynb",
    ],
    "25": [
        "course/Part7_ModernVLLM/1_speculative/part7_spc_1_guessAndCheck.ipynb",
    ],
    "26": [
        "course/Part7_ModernVLLM/3_guided/part7_gdd_1_makeItImpossible.ipynb",
    ],
    "27": [
        "course/Part6_TheServer/1_async/part6_asy_1_theLoopThatMustNotBlock.ipynb",
    ],
    "28": [
        "course/Part8_TheCapstone/3_roof/part8_roof_1_theRatioThatTravels.ipynb",
        "course/Part8_TheCapstone/3_roof/part8_roof_2_CCmeasureYourEngine_helper.ipynb",
    ],
}


def notebooks_for_stage(stage_label: str) -> list[str]:
    return STAGE_NOTEBOOKS.get(stage_label, [])
