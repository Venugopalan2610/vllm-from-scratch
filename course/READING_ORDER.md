# Reading order

Read the notebooks in this order. In each section, read the demos first. Then
do the challenge. Open the solution only after you try the challenge.

`dev/jargon.py` reads this file. It uses the order to find a term that a
notebook uses before the course defines it.

## Part 0, From a Program to a Model

1. `Part0_FromAProgramToAModel/model/part0_mdl_aModelIsAFunction.ipynb`
2. `Part0_FromAProgramToAModel/model/part0_mdl_theGenerationLoop.ipynb`
3. `Part0_FromAProgramToAModel/model/part0_mdl_CCwriteTheLoop_helper.ipynb`
4. `Part0_FromAProgramToAModel/attention/part0_att_attentionIsALookup.ipynb`
5. `Part0_FromAProgramToAModel/attention/part0_att_headsAndTheCache.ipynb`
6. `Part0_FromAProgramToAModel/attention/part0_att_CCmemoizeTheLookup_helper.ipynb`
7. `Part0_FromAProgramToAModel/gpu/part0_gpu_aDifferentKindOfMachine.ipynb`
8. `Part0_FromAProgramToAModel/gpu/part0_gpu_aKernelIsALoopBody.ipynb`
9. `Part0_FromAProgramToAModel/gpu/part0_gpu_CCpredictThenMeasure_helper.ipynb`

## Part 1, The Naive Loop

1. `Part1_TheNaiveLoop/arithmetic/part1_arith_readVsCompute.ipynb`
2. `Part1_TheNaiveLoop/arithmetic/part1_arith_whatTheCacheCosts.ipynb`
3. `Part1_TheNaiveLoop/arithmetic/part1_arith_CCbudgetAServer_helper.ipynb`
4. `Part1_TheNaiveLoop/roofline/part1_roof_measureYourCard.ipynb`
5. `Part1_TheNaiveLoop/roofline/part1_roof_theKnee.ipynb`
6. `Part1_TheNaiveLoop/roofline/part1_roof_CCtwoMachines_helper.ipynb`
7. `Part1_TheNaiveLoop/kvCache/part1_kv_theCostOfForgetting.ipynb`
8. `Part1_TheNaiveLoop/kvCache/part1_kv_theCostOfRemembering.ipynb`
9. `Part1_TheNaiveLoop/kvCache/part1_kv_CCwriteBothLoops_helper.ipynb`

## Part 2, Batching

1. `Part2_Batching/padding/part2_pad_freeSequences.ipynb`
2. `Part2_Batching/padding/part2_pad_theSlotThatWaited.ipynb`
3. `Part2_Batching/padding/part2_pad_CCpickTheBatchSize_helper.ipynb`
4. `Part2_Batching/continuous/part2_cont_iterationLevel.ipynb`
5. `Part2_Batching/continuous/part2_cont_latencyUnderLoad.ipynb`
6. `Part2_Batching/continuous/part2_cont_CCwriteTheScheduler_helper.ipynb`

## Part 3, PagedAttention

1. `Part3_PagedAttention/blocks/part3_blk_fragmentation.ipynb`
2. `Part3_PagedAttention/blocks/part3_blk_thePageTable.ipynb`
3. `Part3_PagedAttention/blocks/part3_blk_CCbuildTheAllocator_helper.ipynb`
4. `Part3_PagedAttention/gather/part3_gat_attentionFromScratch.ipynb`
5. `Part3_PagedAttention/gather/part3_gat_throughThePageTable.ipynb`
6. `Part3_PagedAttention/gather/part3_gat_CCwriteTheOracle_helper.ipynb`
7. `Part3_PagedAttention/kernels/part3_kern_memoryTransactions.ipynb`
8. `Part3_PagedAttention/kernels/part3_kern_fillingTheMachine.ipynb`
9. `Part3_PagedAttention/kernels/part3_kern_CCcoalesceTheGather_helper.ipynb`
10. `Part3_PagedAttention/sharing/part3_shr_copyOnWrite.ipynb`
11. `Part3_PagedAttention/sharing/part3_shr_prefixCache.ipynb`
12. `Part3_PagedAttention/sharing/part3_shr_CCbuildThePrefixCache_helper.ipynb`

## Part 4, The Scheduler

1. `Part4_TheScheduler/admission/part4_adm_runningOutMidFlight.ipynb`
2. `Part4_TheScheduler/admission/part4_adm_swapVsRecompute.ipynb`
3. `Part4_TheScheduler/admission/part4_adm_CCbuildTheScheduler_helper.ipynb`
4. `Part4_TheScheduler/chunked/part4_chk_theStall.ipynb`
5. `Part4_TheScheduler/chunked/part4_chk_CCmixTheBatch_helper.ipynb`

## Part 5, Making It Fast

1. `Part5_MakingItFast/cudaGraphs/part5_cg_thePythonTax.ipynb`
2. `Part5_MakingItFast/cudaGraphs/part5_cg_shapeBuckets.ipynb`
3. `Part5_MakingItFast/cudaGraphs/part5_cg_CCcaptureAndBucket_helper.ipynb`
4. `Part5_MakingItFast/sampling/part5_smp_fourKnobs.ipynb`
5. `Part5_MakingItFast/sampling/part5_smp_oneVectorizedPass.ipynb`
6. `Part5_MakingItFast/sampling/part5_smp_CCsampleItRight_helper.ipynb`
7. `Part5_MakingItFast/detokenize/part5_dtk_mojibake.ipynb`
8. `Part5_MakingItFast/detokenize/part5_dtk_CCstreamItProperly_helper.ipynb`

## Part 6, The Server

1. `Part6_TheServer/async/part6_asy_theLoopThatMustNotBlock.ipynb`
2. `Part6_TheServer/async/part6_asy_CCbuildTheEngine_helper.ipynb`
3. `Part6_TheServer/metrics/part6_met_theNumbersThatMatter.ipynb`
4. `Part6_TheServer/metrics/part6_met_CCwhatIsBroken_helper.ipynb`

## Part 7, Modern vLLM

1. `Part7_ModernVLLM/speculative/part7_spc_guessAndCheck.ipynb`
2. `Part7_ModernVLLM/speculative/part7_spc_CCproveItIsUnbiased_helper.ipynb`
3. `Part7_ModernVLLM/quantization/part7_qnt_halveTheBytes.ipynb`
4. `Part7_ModernVLLM/quantization/part7_qnt_CCwhereDidTheSpeedGo_helper.ipynb`
5. `Part7_ModernVLLM/guided/part7_gdd_makeItImpossible.ipynb`
6. `Part7_ModernVLLM/guided/part7_gdd_CCmaskTheLogits_helper.ipynb`
7. `Part7_ModernVLLM/tensorParallel/part7_tp_columnThenRow.ipynb`
8. `Part7_ModernVLLM/tensorParallel/part7_tp_CCcountTheCollectives_helper.ipynb`

## Part 8, The Capstone

1. `Part8_TheCapstone/engine/part8_eng_theFlatBatch.ipynb`
2. `Part8_TheCapstone/engine/part8_eng_oneRule.ipynb`
3. `Part8_TheCapstone/engine/part8_eng_CCbuildTheScheduler_helper.ipynb`
4. `Part8_TheCapstone/graphs/part8_gr_launchesNotBytes.ipynb`
5. `Part8_TheCapstone/roof/part8_roof_theRatioThatTravels.ipynb`
6. `Part8_TheCapstone/roof/part8_roof_CCmeasureYourEngine_helper.ipynb`
