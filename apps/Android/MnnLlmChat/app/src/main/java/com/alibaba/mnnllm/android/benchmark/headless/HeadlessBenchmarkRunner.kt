package com.alibaba.mnnllm.android.benchmark.headless

import android.content.Context
import android.util.Log
import com.alibaba.mnnllm.android.benchmark.MemoryMonitor
import com.alibaba.mnnllm.android.llm.ChatService
import com.alibaba.mnnllm.android.llm.GenerateProgressListener
import com.alibaba.mnnllm.android.llm.LlmSession
import com.alibaba.mnnllm.android.llm.SamplerOverrides
import java.io.File
import java.util.concurrent.atomic.AtomicLong

/**
 * Runs one or more headless inference requests against a SINGLE loaded
 * [LlmSession] and logs TTFT/prefill/decode/RSS/power/thermal metrics per
 * generation, tagged by run_id (or, for a multi-generation sweep, by a
 * per-generation sub run_id) to Logcat.
 *
 * Deliberately does NOT go through
 * [com.alibaba.mnnllm.api.openai.runtime.LlmRuntimeController]'s cached
 * getActiveSession(): a headless run owns its own [LlmSession] end to end -
 * created via [ChatService.createLlmSession], loaded, run, and fully
 * released - so that a peak-RSS reading for one model file is never
 * contaminated by a larger model left resident from a previous headless or
 * chat-UI session in the same process.
 *
 * [warmupRuns]/[trials] (both default to the original single-shot
 * behavior: 0 warmups, 1 trial) let a single broadcast run an entire
 * warmup+trials sweep against ONE loaded session instead of a fresh
 * session per generation. This matters specifically for GPU backends:
 * OpenCL kernels are JIT-compiled by the driver lazily, on the first
 * actual forward pass through each op/shape - not at load() time - and
 * that compile cost is amortized only across calls that share the same
 * live session/runtime. A fresh LlmSession per trial (the original
 * design, still what [warmupRuns]=0,[trials]=1 gives you per broadcast)
 * pays that cold-compile cost on literally every single trial, including
 * the "warmup" one, which is indistinguishable from every other
 * catastrophically-slow trial. Looping generations inside one session
 * instead matches the paper methodology this harness is trying to
 * reproduce ("one warm-up run followed by at least three recorded
 * trials") and lets a real warmup actually warm something up.
 */
class HeadlessBenchmarkRunner(private val context: Context) {

    fun run(
        modelPath: String,
        prompt: String,
        runId: String,
        maxTokens: Int = BenchmarkHeadlessReceiver.DEFAULT_MAX_TOKENS,
        topK: Int? = null,
        topP: Float? = null,
        minP: Float? = null,
        backendType: String? = null,
        warmupRuns: Int = 0,
        trials: Int = 1
    ) {
        val powerSampler = PowerSampler(context)
        val thermalSampler = ThermalSampler(context)
        var session: LlmSession? = null

        try {
            val configPath = "$modelPath/config.json"
            if (!File(configPath).exists()) {
                logError(runId, "config_not_found", "No config.json under $modelPath")
                return
            }
            // Namespaced away from real catalog model IDs so this can never
            // collide with (or pick up leftover per-model state from) a
            // model the user has loaded through the normal chat UI.
            val modelId = "headless/$modelPath"

            // RSS is tracked across the full call - load through every
            // generation in this sweep - since model load is frequently
            // where peak RSS actually occurs, and the peak only ever grows
            // monotonically across a sweep of same-model generations.
            MemoryMonitor.reset()
            MemoryMonitor.start(intervalSeconds = 1)

            // null unless the caller explicitly provided at least one of
            // top_k/top_p/min_p - when none are provided this is null and
            // load() applies zero overrides, leaving the model's normal
            // shipped sampler config completely untouched. Must be built
            // and passed in BEFORE createLlmSession()/load(): MNN's Sampler
            // is constructed once at load time and never re-reads config
            // afterward, so there is no post-load equivalent for these
            // (see SamplerOverrides' kdoc).
            val samplerOverrides = if (topK != null || topP != null || minP != null) {
                SamplerOverrides(topK = topK, topP = topP, minP = minP)
            } else null
            Log.i(LOG_TAG, "run_id=$runId SAMPLER_OVERRIDE_TOP_K=${topK ?: "default"} " +
                "SAMPLER_OVERRIDE_TOP_P=${topP ?: "default"} SAMPLER_OVERRIDE_MIN_P=${minP ?: "default"}")

            Log.i(LOG_TAG, "run_id=$runId BACKEND_TYPE_OVERRIDE=${backendType ?: "default"}")
            Log.i(LOG_TAG, "run_id=$runId SWEEP_CONFIG warmup_runs=$warmupRuns trials=$trials")
            val coldLoadStartMs = System.currentTimeMillis()
            val newSession = ChatService.provide().createLlmSession(
                modelId,
                configPath,
                "headless_${runId}_${System.currentTimeMillis()}",
                null,
                false,
                backendType = backendType,
                samplerOverrides = samplerOverrides
            )
            session = newSession
            // Force single-turn isolation before load(): every headless call
            // must be a genuinely isolated request with zero prior
            // conversation history, no matter how many headless calls ran
            // earlier in this process. The session is also freshly created
            // with no history list, so this is belt-and-suspenders - but
            // setting it explicitly (and before load, since it feeds the
            // native extra_config at construction time) means correctness
            // here never depends on some other default staying what it is
            // today. keep_history=false also makes every generation below
            // start from the same clean single-turn state (see Response()
            // in llm_session.cpp: history_.resize(1) when !keep_history_),
            // so looping multiple generations in one session never grows
            // the effective prompt across trials.
            newSession.setKeepHistory(false)
            newSession.load()
            val coldLoadMs = System.currentTimeMillis() - coldLoadStartMs

            if (!newSession.isModelLoaded()) {
                logError(runId, "model_load_failed", "LlmSession reported not loaded after load() for $modelPath")
                return
            }

            // MemoryMonitor.start()'s own guaranteed sample fires before this
            // point (it has to - it's called before createLlmSession()/load()
            // above, so the timer is running throughout load), so it only
            // ever captures pre-load baseline RSS. Force one more guaranteed
            // sample right here, with weights now fully resident, so a
            // request that finishes generation faster than one polling tick
            // still has a real post-load reading rather than reporting that
            // pre-load baseline as if it were the peak.
            MemoryMonitor.sampleNow()

            // Disable thinking mode at the template level (the same
            // "enable_thinking" mechanism HuggingFace's
            // apply_chat_template(enable_thinking=False) uses), not the
            // text-based /no_think trick - confirmed unreliable on
            // Qwen3.5. persist=false: this is a synthetic, throwaway
            // session and must stay otherwise stateless, so this pushes
            // straight to native config without writing a
            // custom_config.json sidecar under the headless modelId.
            newSession.updateThinking(false, persist = false)
            Log.i(LOG_TAG, "run_id=$runId ENABLE_THINKING=false")

            // Hard cap on decode length: without this, a stuck/repeating
            // model has no lower ceiling than whatever config.json (or the
            // native 2048 default) specifies - confirmed to let a single
            // headless call run for minutes generating garbage. This is set
            // per-call, right before generate(), rather than baked into the
            // model config, so it only affects headless runs.
            newSession.updateMaxNewTokens(maxTokens)
            Log.i(LOG_TAG, "run_id=$runId MAX_TOKENS=$maxTokens")

            var isFirstGeneration = true
            for (w in 1..warmupRuns) {
                val subRunId = "${runId}_w$w"
                Log.i(LOG_TAG, "run_id=$runId SWEEP_STEP phase=warmup index=$w sub_run_id=$subRunId")
                runOneGeneration(
                    newSession, prompt, subRunId, powerSampler, thermalSampler,
                    coldLoadMs = if (isFirstGeneration) coldLoadMs else null,
                    loadStartEpochMs = if (isFirstGeneration) coldLoadStartMs else null
                )
                isFirstGeneration = false
            }
            for (t in 1..trials) {
                val subRunId = "${runId}_t$t"
                Log.i(LOG_TAG, "run_id=$runId SWEEP_STEP phase=trial index=$t sub_run_id=$subRunId")
                runOneGeneration(
                    newSession, prompt, subRunId, powerSampler, thermalSampler,
                    coldLoadMs = if (isFirstGeneration) coldLoadMs else null,
                    loadStartEpochMs = if (isFirstGeneration) coldLoadStartMs else null
                )
                isFirstGeneration = false
            }
        } catch (e: Exception) {
            Log.e(LOG_TAG, "run_id=$runId unhandled exception", e)
            logError(runId, "exception", e.message ?: e.toString())
        } finally {
            powerSampler.stop()
            thermalSampler.stop()
            MemoryMonitor.stop()
            // Always fully release - never leave this session cached for
            // reuse. Reuse across different model files (or across a
            // process's separate headless calls) is exactly what would
            // contaminate the next run's peak-RSS reading; reuse WITHIN one
            // sweep's own warmup+trials loop above is fine since it's
            // always the same model/backend/config for the whole sweep.
            try {
                session?.release()
            } catch (e: Exception) {
                Log.w(LOG_TAG, "run_id=$runId failed to release session: ${e.message}")
            }
        }
    }

    /**
     * Runs exactly one generation against an already-loaded [session] and
     * logs the same COLD_LOAD_MS/TTFT_MS/.../RUN_DONE tag lines the original
     * single-shot run() always logged - just under [subRunId] instead of
     * the sweep's outer run_id, so run_mnn_autobench.py's existing
     * run_id-substring log parsing (parse_run()/tag_of(), unchanged) can
     * pull out each generation's block independently from one shared
     * logcat capture. [coldLoadMs] is only non-null for the very first
     * generation in a sweep (the only one a real load() preceded); passing
     * null omits COLD_LOAD_MS entirely for every later generation, which
     * Python already treats as "not reported" (see build_metrics()) rather
     * than a misleading 0ms reload.
     */
    private fun runOneGeneration(
        session: LlmSession,
        prompt: String,
        subRunId: String,
        powerSampler: PowerSampler,
        thermalSampler: ThermalSampler,
        coldLoadMs: Long?,
        loadStartEpochMs: Long? = null
    ) {
        val ttftCapturedNanos = AtomicLong(-1)
        val requestStartNanos = System.nanoTime()
        // Wall-clock markers (epoch ms) logged as tags below so the host harness can compute TTLT,
        // decode rate and energy windows without any per-chunk logging. OPLUS logd drops a process's
        // log lines beyond ~300 rows per second, and one line per streamed chunk overflowed that,
        // silently dropping the result tags (the harness then waited forever).
        val dispatchEpochMs = System.currentTimeMillis()
        var firstChunkEpochMs = -1L
        var lastChunkEpochMs = -1L
        var streamChunks = 0
        val responseBuilder = StringBuilder()

        // Power/thermal are sampled only across the inference window itself
        // (not model load), per spec - restarted fresh for every generation
        // in the sweep so each trial gets its own independent reading.
        powerSampler.start(intervalMs = 100)
        thermalSampler.start(intervalMs = 1000)

        val result = session.generate(prompt, emptyMap(), object : GenerateProgressListener {
            override fun onProgress(progress: String?): Boolean {
                if (progress != null) {
                    // Wall-clock TTFT: time from submit to the first
                    // actual streamed token, captured in Kotlin. This is
                    // intentionally separate from the native prefill_time
                    // below - it also captures JNI dispatch/scheduling
                    // overhead that native compute time alone misses.
                    ttftCapturedNanos.compareAndSet(-1L, System.nanoTime() - requestStartNanos)
                    val now = System.currentTimeMillis()
                    if (firstChunkEpochMs < 0) firstChunkEpochMs = now
                    lastChunkEpochMs = now
                    streamChunks++
                    responseBuilder.append(progress)
                }
                if (progress != null) {
                    // Additional safety net alongside the token cap above:
                    // a stuck model can loop on the same short substring
                    // (observed: "...\n\n\n" repeating) well before
                    // max_tokens is even reached. Returning true here
                    // requests early stop the same way user-cancel does -
                    // generation halts cleanly and whatever was decoded
                    // so far is still returned normally below.
                    if (hasRepeatingTail(responseBuilder, REPEAT_WINDOW_CHARS, REPEAT_MIN_REPEATS, subRunId)) {
                        Log.i(
                            LOG_TAG,
                            "run_id=$subRunId REPETITION_DETECTED stopping early responseLength=${responseBuilder.length}"
                        )
                        return true
                    }
                }
                return false // never request early stop otherwise for a benchmark run
            }
        })
        powerSampler.stop()
        thermalSampler.stop()

        val ttftMs = if (ttftCapturedNanos.get() >= 0) ttftCapturedNanos.get() / 1_000_000.0 else -1.0
        val promptLen = result["prompt_len"] as? Long ?: 0L
        val decodeLen = result["decode_len"] as? Long ?: 0L
        val prefillTimeUs = result["prefill_time"] as? Long ?: 0L
        val decodeTimeUs = result["decode_time"] as? Long ?: 0L

        val peakRssKb = MemoryMonitor.getMaxMemoryPssKb()
        val avgPowerMa = powerSampler.getAverageMa()
        val energyMasSampled = powerSampler.getEnergyMasSampled()
        val energyMjSampled = powerSampler.getEnergyMjSampled()
        val cpuTempC = thermalSampler.getMaxCpuTempC()
        val skinTempC = thermalSampler.getMaxSkinTempC()
        val thermalStatus = thermalSampler.getThermalStatusLabel()

        if (coldLoadMs != null) {
            Log.i(LOG_TAG, "run_id=$subRunId COLD_LOAD_MS=$coldLoadMs")
        }
        Log.i(LOG_TAG, "run_id=$subRunId TTFT_MS=$ttftMs")
        Log.i(LOG_TAG, "run_id=$subRunId DISPATCH_EPOCH_MS=$dispatchEpochMs")
        Log.i(LOG_TAG, "run_id=$subRunId FIRST_TOKEN_EPOCH_MS=${if (firstChunkEpochMs >= 0) firstChunkEpochMs else "unavailable"}")
        Log.i(LOG_TAG, "run_id=$subRunId LAST_TOKEN_EPOCH_MS=${if (lastChunkEpochMs >= 0) lastChunkEpochMs else "unavailable"}")
        Log.i(LOG_TAG, "run_id=$subRunId STREAM_CHUNKS=$streamChunks")
        if (loadStartEpochMs != null) {
            Log.i(LOG_TAG, "run_id=$subRunId LOAD_START_EPOCH_MS=$loadStartEpochMs")
        }
        Log.i(LOG_TAG, "run_id=$subRunId PREFILL_TIME_US=$prefillTimeUs")
        Log.i(LOG_TAG, "run_id=$subRunId DECODE_TIME_US=$decodeTimeUs")
        Log.i(LOG_TAG, "run_id=$subRunId PROMPT_LEN=$promptLen")
        Log.i(LOG_TAG, "run_id=$subRunId DECODE_LEN=$decodeLen")
        Log.i(LOG_TAG, "run_id=$subRunId PEAK_RSS_KB=$peakRssKb")
        Log.i(LOG_TAG, "run_id=$subRunId POWER_MA=${avgPowerMa?.let { "%.2f".format(it) } ?: "unavailable"}")
        Log.i(LOG_TAG, "run_id=$subRunId ENERGY_MAS_SAMPLED=${energyMasSampled?.let { "%.3f".format(it) } ?: "unavailable"}")
        Log.i(LOG_TAG, "run_id=$subRunId ENERGY_MJ_SAMPLED=${energyMjSampled?.let { "%.3f".format(it) } ?: "unavailable"}")
        Log.i(LOG_TAG, "run_id=$subRunId THERMAL_STATUS=$thermalStatus")
        Log.i(LOG_TAG, "run_id=$subRunId THERMAL_TEMP_CPU_C=${cpuTempC?.let { "%.1f".format(it) } ?: "unavailable"}")
        Log.i(LOG_TAG, "run_id=$subRunId THERMAL_TEMP_SKIN_C=${skinTempC?.let { "%.1f".format(it) } ?: "unavailable"}")
        Log.i(LOG_TAG, "run_id=$subRunId RUN_DONE response=${escapeForSingleLineLog(responseBuilder.toString())}")
    }

    private fun logError(runId: String, reason: String, message: String) {
        Log.i(LOG_TAG, "run_id=$runId RUN_ERROR reason=$reason message=$message")
    }

    /**
     * Android's Log.i()/logcat does not render an embedded newline as part
     * of one flat line - it splits the entry, so anything after the first
     * `\n` in a raw model response either shows as an unprefixed
     * continuation line or is invisible to a single-line viewer/regex-based
     * parser. Since real LLM output routinely starts with or contains
     * newlines, escape control characters so the full response always
     * survives as one physical logcat line.
     */
    private fun escapeForSingleLineLog(text: String): String {
        return text
            .replace("\\", "\\\\")
            .replace("\n", "\\n")
            .replace("\r", "\\r")
    }

    /**
     * True if the last (windowSize * minRepeats) characters of [text]
     * consist of the same [windowSize]-character block repeated [minRepeats]
     * times in a row. A rigid, fixed-window check rather than a general
     * periodicity detector - deliberately simple, just enough to catch a
     * model stuck emitting the same short chunk (e.g. "...\n\n\n") over and
     * over.
     */
    // TEMPORARY: runId param + REPDEBUG logging added purely to observe why
    // this isn't triggering on a confirmed-repetitive real response. No
    // change to window size, repeat count, comparison logic, or control
    // flow (still returns false on the first non-matching block, same as
    // before) - remove alongside the rest of the REPDEBUG logging once the
    // root cause is confirmed.
    private fun hasRepeatingTail(text: CharSequence, windowSize: Int, minRepeats: Int, runId: String): Boolean {
        val totalLen = windowSize * minRepeats
        if (text.length < totalLen) {
            Log.i("REPDEBUG", "run_id=$runId hasRepeatingTail SKIPPED textLength=${text.length} totalLenNeeded=$totalLen")
            return false
        }
        val tailStart = text.length - totalLen
        val unit = text.substring(tailStart, tailStart + windowSize)
        for (i in 1 until minRepeats) {
            val blockStart = tailStart + i * windowSize
            val block = text.substring(blockStart, blockStart + windowSize)
            val blockMatches = block == unit
            Log.i(
                "REPDEBUG",
                "run_id=$runId hasRepeatingTail textLength=${text.length} tailStart=$tailStart i=$i " +
                    "unit=\"${escapeForSingleLineLog(unit)}\" block=\"${escapeForSingleLineLog(block)}\" matches=$blockMatches"
            )
            if (!blockMatches) {
                return false
            }
        }
        Log.i(
            "REPDEBUG",
            "run_id=$runId hasRepeatingTail TRIGGERED textLength=${text.length} unit=\"${escapeForSingleLineLog(unit)}\""
        )
        return true
    }

    companion object {
        const val LOG_TAG = "BenchmarkHeadless"
        private const val REPEAT_WINDOW_CHARS = 16
        private const val REPEAT_MIN_REPEATS = 5
    }
}
