// Created by ruoyi.sjd on 2025/5/7.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.llm;

import android.util.Log
import com.alibaba.mnnllm.android.llm.ChatService.Companion.provide
import com.alibaba.mnnllm.android.chat.model.ChatDataItem
import com.alibaba.mnnllm.android.modelsettings.ModelConfig
import com.alibaba.mnnllm.android.model.ModelTypeUtils
import com.alibaba.mnnllm.android.modelsettings.ModelConfig.Companion.getExtraConfigFile
import com.google.gson.Gson
import com.google.gson.JsonObject
import timber.log.Timber
import java.io.File
import java.util.stream.Collectors
import kotlin.concurrent.Volatile
import android.util.Pair
import com.alibaba.mnnllm.android.utils.MmapUtils
import android.content.Context
import android.app.ActivityManager
import com.alibaba.mnnllm.android.modelsettings.Jinja
import com.alibaba.mnnllm.android.modelsettings.JinjaContext
import com.alibaba.mnnllm.android.modelsettings.ModelConfig.Companion.loadConfig
import com.alibaba.mnnllm.android.utils.FileSplitter
import com.alibaba.mnnllm.android.qnn.QnnModule

/**
 * Optional per-session sampler-parameter overrides, applied in-memory to
 * the config JSON before load()/initNative() - never persisted to disk.
 *
 * This exists specifically because the native Sampler object is
 * constructed exactly once, inside Llm::load(), from whatever config
 * values exist at that moment (transformers/llm/engine/src/llm.cpp:330,
 * mSampler.reset(Sampler::createSampler(mContext, mConfig))) and is never
 * rebuilt by a later set_config() call. Unlike enable_thinking - which IS
 * re-applied live via setChatTemplate() on every set_config() call, so a
 * post-load updateConfig() push works - these values MUST already be
 * present before the initial load()/set_config() sequence to have any
 * effect at all. Pushing them afterward via updateConfig() is a silent
 * no-op: it updates the stored config JSON but the already-built Sampler
 * never re-reads it.
 */
data class SamplerOverrides(
    val topK: Int? = null,
    val topP: Float? = null,
    val minP: Float? = null
)

class LlmSession (
    private val modelId: String,
    override var sessionId: String,
    private val configPath: String,
    var savedHistory: List<ChatDataItem>?,
    var backendType: String? = null,
    private val useCustomConfig: Boolean = true,
    private val samplerOverrides: SamplerOverrides? = null
): ChatSession{
    override var supportOmni: Boolean = false
    private var nativePtr: Long = 0

    @Volatile
    private var modelLoading = false

    @Volatile
    private var generating = false

    @Volatile
    private var releaseRequested = false

    private var keepHistory = false

    private var isQnn = false

    override fun getHistory(): List<ChatDataItem>?{
        return savedHistory
    }

    override fun setHistory(history: List<ChatDataItem>?) {
    }

    override fun load() {
        Log.d(TAG, "MNN_DEBUG load begin modelId: $modelId backend: $backendType")
        modelLoading = true
        isQnn = ModelTypeUtils.isQnnModel(modelId)

        checkAndMergeSplitFiles()
        var historyStringList: List<String>? = null
        val currentHistory = this.savedHistory
        if (!currentHistory.isNullOrEmpty()) {
            historyStringList =
                    currentHistory.stream()
                            .map { obj: ChatDataItem -> obj.text }
                    .filter { obj: String? -> obj != null }
                    .map { obj: String? -> obj!! }
                    .collect(Collectors.toList())
        }
        val config = if (useCustomConfig) {
            ModelConfig.loadMergedConfig(configPath, getExtraConfigFile(modelId))!!
        } else {
            ModelConfig.loadDefaultConfig(configPath)!!
        }
        var rootCacheDir: String? = ""
        if (config.useMmap == true) {
            rootCacheDir = MmapUtils.getMmapDir(modelId)
            File(rootCacheDir).mkdirs()
        }
        val configMap = HashMap<String, Any>().apply {
            put("is_r1", ModelTypeUtils.isR1Model(modelId))
            put("mmap_dir", rootCacheDir ?: "")
            put("keep_history", keepHistory)
        }
        val llmConfig = if (useCustomConfig) {
            ModelConfig.loadMergedConfig(configPath, getExtraConfigFile(modelId))!!
        } else {
            ModelConfig.loadDefaultConfig(configPath)!!
        }
        // Override backend type from constructor only if not null
        if (backendType != null) {
            llmConfig.backendType = backendType
        }
        if (isQnn) {
            llmConfig.visualModel = "visual_qnn_${QnnModule.modelMiddleName()}.mnn"
        }
        // Applied here, before serialization/initNative(), since this is
        // the only point that actually affects the native Sampler - see
        // SamplerOverrides' kdoc for why a post-load updateConfig() push
        // would silently do nothing. Only overrides fields explicitly
        // provided; anything left null keeps the model's normal config
        // untouched, same as if no override were passed at all.
        samplerOverrides?.let { overrides ->
            overrides.topK?.let { llmConfig.topK = it }
            overrides.topP?.let { llmConfig.topP = it }
            overrides.minP?.let { llmConfig.minP = it }
            Log.i(
                TAG,
                "MNN_DEBUG applying sampler overrides: topK=${overrides.topK} topP=${overrides.topP} minP=${overrides.minP}"
            )
        }
        Log.d(TAG, "MNN_DEBUG load initNative")
        nativePtr = initNative(
                configPath,
                historyStringList,
        if (llmConfig != null) {
            Gson().toJson(llmConfig)
        } else {
            "{}"
        },
        Gson().toJson(configMap)
        )
        Log.d(TAG, "MNN_DEBUG load initNative end")
        modelLoading = false
        if (nativePtr == 0L) {
            Log.e(TAG, "Model load failed - native initialization returned null pointer")
            throw IllegalStateException("Model load failed - the model module could not be loaded")
        }
        if (releaseRequested) {
            release()
        }
    }

    /**
     * Check if the model is successfully loaded and ready for inference
     */
    fun isModelLoaded(): Boolean {
        return nativePtr != 0L
    }
    
    /**
     * Check and merge split files for the current model
     */
    private fun checkAndMergeSplitFiles() {
        try {
            val configFile = File(configPath)
            val modelDir = configFile.parentFile
            
            if (modelDir != null && modelDir.exists()) {
                Log.d(TAG, "Checking for split files in model directory: ${modelDir.absolutePath}")
                
                if (FileSplitter.needsMerging(modelDir)) {
                    Log.d(TAG, "Found split files that need merging in ${modelDir.absolutePath}")
                    val success = FileSplitter.mergeAllSplitFiles(modelDir)
                    if (success) {
                        Log.d(TAG, "Successfully merged split files for model: $modelId")
                    } else {
                        Log.w(TAG, "Failed to merge some split files for model: $modelId")
                    }
                } else {
                    Log.d(TAG, "No split files found for model: $modelId")
                }
            } else {
                Log.w(TAG, "Model directory not found: ${modelDir?.absolutePath}")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error checking/merging split files for model: $modelId", e)
        }
    }

    fun getConfig(): ModelConfig? {
        return ModelConfig.loadMergedConfig(configPath, getExtraConfigFile(modelId))
    }

    private fun generateNewSessionId(): String {
        this.sessionId = System.currentTimeMillis().toString()
        return this.sessionId
    }

    override fun generate(prompt: String,
                          params: Map<String, Any>,
                          progressListener: GenerateProgressListener): HashMap<String, Any> {
        Log.d(TAG, "start generate prompt: $prompt")
        synchronized(this) {
            if (mockLatex) {
                Timber.d("MNN_DEBUG generate intercepted by mockLatex")
                return submitMockLatexHistory(progressListener)
            }
            Log.d(TAG, "MNN_DEBUG submit$prompt")
            generating = true
            val result = submitNative(nativePtr, prompt, keepHistory, progressListener)
            generating = false
            if (releaseRequested) {
                release()
            }
            return result
        }
    }

    override fun reset(): String {
        synchronized(this) {
            resetNative(nativePtr)
        }
        return generateNewSessionId()
    }

    override fun release() {
        synchronized(this) {
            Log.d(
                    TAG,
                    "MNN_DEBUG release nativePtr: $nativePtr mGenerating: $generating"
            )
            if (!generating && !modelLoading) {
                releaseInner()
            } else {
                releaseRequested = true
                while (generating || modelLoading) {
                    try {
                        (this as Object).wait()
                    } catch (e: InterruptedException) {
                        Thread.currentThread().interrupt()
                        Log.e(TAG, "Thread interrupted while waiting for release", e)
                    }
                }
                releaseInner()
            }
        }
    }

    private fun releaseInner() {
        if (nativePtr != 0L) {
            releaseNative(nativePtr)
            nativePtr = 0
            provide().removeSession(sessionId)
            (this as Object).notifyAll()
        }
    }

    private external fun initNative(
            configPath: String?,
            history: List<String>?,
            mergedConfigStr: String?,
            configJsonStr: String?
    ): Long

    private external fun submitNative(
            instanceId: Long,
            input: String,
            keepHistory: Boolean,
            listener: GenerateProgressListener
    ): HashMap<String, Any>

    private external fun resetNative(instanceId: Long)

    private external fun getDebugInfoNative(instanceId: Long): String

    private external fun releaseNative(instanceId: Long)

    private external fun setWavformCallbackNative(
            instanceId: Long,
            listener: AudioDataListener?
    ): Boolean

    override fun setKeepHistory(keepHistory: Boolean) {
        this.keepHistory = keepHistory
    }

    override fun setEnableAudioOutput(enable: Boolean) {
        updateEnableAudioOutputNative(nativePtr, enable)
    }

    override val debugInfo
        get() = getDebugInfoNative(nativePtr) + "\n"


    fun setAudioDataListener(listener: AudioDataListener?) {
        synchronized(this) {
            if (nativePtr != 0L) {
                setWavformCallbackNative(nativePtr, listener)
            } else {
                Log.e(TAG, "nativePtr null")
            }
        }
    }

    fun updateMaxNewTokens(maxNewTokens: Int) {
        updateMaxNewTokensNative(nativePtr, maxNewTokens)
    }

    fun updateSystemPrompt(systemPrompt: String) {
        updateSystemPromptNative(nativePtr, systemPrompt)
    }

    override fun updateThinking(thinking: Boolean) {
        updateThinking(thinking, persist = true)
    }

    /**
     * Same as [updateThinking] but when [persist] is false, skips both the
     * loadConfig(modelId) lookup/merge and the custom_config.json disk
     * write, pushing just the jinja fragment to native directly.
     *
     * loadConfig(modelId) resolves the model's config file via
     * ModelConfig.getDefaultConfigFile(), which only understands "local/",
     * "Builtin/", and catalog-downloaded model IDs. A synthetic, non-catalog
     * modelId (e.g. the headless benchmark path's "headless/$modelPath")
     * matches none of those and getDefaultConfigFile returns null, so
     * loadConfig(modelId) returns null and the persist=true path's
     * `loadedConfig?.let { ... }` would silently no-op - thinking mode
     * would never actually change. Bypassing that lookup avoids depending
     * on it resolving at all, and keeps each headless call otherwise
     * stateless (no custom_config.json sidecar left behind under its
     * synthetic modelId), matching the RSS-isolation design principle: a
     * fresh LlmSession per call, released afterward, nothing cached.
     *
     * Native set_config() merges the pushed JSON at the top level
     * (mls::LlmSession::updateConfig in llm_session.cpp), so pushing just
     * the "jinja" key is sufficient - it neither requires nor clobbers the
     * rest of the model's config.
     */
    fun updateThinking(thinking: Boolean, persist: Boolean) {
        if (persist) {
            val loadedConfig = loadConfig(modelId)
            loadedConfig?.let {
                loadedConfig.jinja = Jinja(context = JinjaContext(enableThinking = thinking))
                ModelConfig.saveConfig(getExtraConfigFile(modelId), loadedConfig)
                updateConfig(Gson().toJson(loadedConfig))
            }
        } else {
            val jinjaFragment = JsonObject()
            jinjaFragment.add("jinja", Gson().toJsonTree(Jinja(context = JinjaContext(enableThinking = thinking))))
            val configJson = jinjaFragment.toString()
            updateConfig(configJson)
            // TEMPORARY: tracing the enable_thinking push from Kotlin through
            // to the native Jinja context. Remove once confirmed working.
            Log.i("THINKDEBUG", "pushed jinja fragment: $configJson")
        }
    }

    fun updateConfig(configJson: String) {
        Log.d(TAG, "updateConfig: $configJson")
        updateConfigNative(nativePtr, configJson)
    }

    private external fun updateEnableAudioOutputNative(llmPtr: Long, enable: Boolean)


    private external fun updateMaxNewTokensNative(llmPtr: Long, maxNewTokens: Int)

    private external fun updateSystemPromptNative(llmPtr: Long, systemPrompt: String)

    private external fun updateAssistantPromptNative(llmPtr: Long, assistantPrompt: String)

    private external fun updateConfigNative(llmPtr: Long, configJson: String)


    companion object {
        const val TAG: String = "LlmSession"
        var mockLatex: Boolean = false
        var mockLatexContent: String? = null

        init {
            System.loadLibrary("mnnllmapp")
        }
    }



    //New: public method supporting complete history messages
    fun submitFullHistory(
        history: List<Pair<String, String>>,
        progressListener: GenerateProgressListener
    ): HashMap<String, Any> {
        synchronized(this) {
            if (mockLatex) {
                Timber.d("MNN_DEBUG submitFullHistory intercepted by mockLatex")
                return submitMockLatexHistory(progressListener)
            }
            //Use Timber instead of Log
            Timber.d("MNN_DEBUG submitFullHistory with ${history.size} messages")
            //Type conversion: kotlin.Pair -> android.util.Pair
            val androidHistory = history.map { android.util.Pair(it.first, it.second) }
            //Call JNI method, remove unnecessary type conversion
            val result = submitFullHistoryNative(nativePtr, androidHistory, progressListener)
            generating = false
            return result
        }
    }

    private fun submitMockLatexHistory(progressListener: GenerateProgressListener): HashMap<String, Any> {
        val mockText = mockLatexContent ?: "Here is a math formula:\n\n\$E=mc^2$\n\nAnd a block formula:\n\n\$\$a^2 + b^2 = c^2\$\$\n\nEnd of mock."
        Thread {
            try {
                // Simulate streaming delay
                var index = 0
                val chunkSize = 3
                while (index < mockText.length) {
                    Thread.sleep(50)
                    val endIndex = Math.min(index + chunkSize, mockText.length)
                    val chunk = mockText.substring(index, endIndex)
                    if (progressListener.onProgress(chunk)) {
                        break
                    }
                    index = endIndex
                }
                progressListener.onProgress(null) // notify completion
            } catch (e: Exception) {
                Timber.e(e, "Mock generation failed")
            } finally {
                generating = false
            }
        }.start()
        val map = HashMap<String, Any>()
        map["success"] = true
        map["prompt_len"] = 10L
        map["decode_len"] = mockText.length.toLong()
        map["prefill_time"] = 100000L
        map["decode_time"] = 2000000L
        return map
    }
    private external fun submitFullHistoryNative(
        nativePtr: Long,
        history: List<android.util.Pair<String, String>>,
        progressListener: GenerateProgressListener
    ): HashMap<String, Any>

    fun modelId(): String {
        //Create temporary variable to avoid modifying original modelId
        return modelId

    }

    fun getSystemPrompt(): String? {
        return getSystemPromptNative(nativePtr)
    }

    private external fun getSystemPromptNative(llmPtr: Long): String?

    private external fun dumpConfigNative(llmPtr: Long): String

    fun dumpConfig(): String {
        return if (nativePtr != 0L) {
            dumpConfigNative(nativePtr)
        } else {
            "{}"
        }
    }

    // Helper function to get current memory usage in MB
    private fun getCurrentMemoryUsageMB(context: Context): Long {
        val runtime = Runtime.getRuntime()
        val usedMemoryBytes = runtime.totalMemory() - runtime.freeMemory()
        return usedMemoryBytes / (1024 * 1024) // Convert to MB
    }
    
    // Helper function to get total memory info
    private fun getMemoryInfo(context: Context): Pair<Long, Long> {
        val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val memoryInfo = ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memoryInfo)
        
        val runtime = Runtime.getRuntime()
        val usedMemoryMB = (runtime.totalMemory() - runtime.freeMemory()) / (1024 * 1024)
        val availMemoryMB = memoryInfo.availMem / (1024 * 1024)
        
        return Pair(usedMemoryMB, availMemoryMB)
    }

    // Official benchmark functionality following llm_bench.cpp approach
    fun runBenchmark(
        context: Context,
        commandParams: com.alibaba.mnnllm.android.benchmark.CommandParameters,
        testInstance: com.alibaba.mnnllm.android.benchmark.TestInstance,
        callback: com.alibaba.mnnllm.android.benchmark.BenchmarkCallback
    ): com.alibaba.mnnllm.android.benchmark.BenchmarkResult {
        // Use coroutine instead of Thread for better lifecycle management
        return try {
            // Run the actual benchmark in C++ following llm_bench.cpp structure
            runBenchmarkNative(
                nativePtr, 
                commandParams.backend,
                commandParams.threads,
                commandParams.useMmap,
                commandParams.power,
                commandParams.precision,
                commandParams.memory,
                commandParams.dynamicOption,
                commandParams.nPrompt,
                commandParams.nGenerate,
                commandParams.nRepeat,
                commandParams.kvCache == "true",
                testInstance,
                callback
            )
        } catch (e: Exception) {
            com.alibaba.mnnllm.android.benchmark.BenchmarkResult(
                testInstance = testInstance,
                success = false,
                errorMessage = "benchmark failed: ${e.message}"
            )
        }
    }

    // C++ implementation following llm_bench.cpp approach
    private external fun runBenchmarkNative(
        nativePtr: Long,
        backend: Int,
        threads: Int,
        useMmap: Boolean,
        power: Int,
        precision: Int,
        memory: Int,
        dynamicOption: Int,
        nPrompt: Int,
        nGenerate: Int,
        nRepeat: Int,
        kvCache: Boolean,
        testInstance: com.alibaba.mnnllm.android.benchmark.TestInstance,
        callback: com.alibaba.mnnllm.android.benchmark.BenchmarkCallback
    ): com.alibaba.mnnllm.android.benchmark.BenchmarkResult

}