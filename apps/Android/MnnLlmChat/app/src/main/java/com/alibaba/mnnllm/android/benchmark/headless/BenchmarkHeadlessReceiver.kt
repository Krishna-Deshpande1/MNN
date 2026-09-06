package com.alibaba.mnnllm.android.benchmark.headless

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.util.Log

/**
 * ADB-reachable entry point for headless benchmarking:
 *   adb shell am broadcast -a com.mnnllmchat.RUN_PROMPT \
 *     -p com.alibaba.mnnllm.android \
 *     --es model_path /data/local/tmp/mnn_models/<model> \
 *     --es prompt "..." --es run_id <id> \
 *     --ei max_tokens 4096 \
 *     --ei top_k 40 --ef top_p 0.9 --ef min_p 0.05
 *
 * `max_tokens` is optional (defaults to [DEFAULT_MAX_TOKENS]) and caps how
 * many tokens a single headless call may decode, so one question can never
 * run unboundedly long regardless of model size or a repetition loop -
 * confirmed empirically: Qwen3-1.7B with /no_think generated ~1900+ tokens
 * of repeating garbage before hitting the native 2048-token ceiling, taking
 * 456s and reaching 81.8C CPU. The default was raised from 80 to 256 after
 * confirming 80 cut off genuine multi-step reasoning (e.g. math word
 * problems) mid-thought before reaching an answer, and again from 256 to
 * 4096 to match run_mnn_autobench.py's own default exactly, so a manual
 * broadcast without this extra (or any other caller that omits it) behaves
 * the same as the Python automation's default rather than silently
 * diverging from it.
 *
 * `top_k`/`top_p`/`min_p` are all optional and independent - omit any (or
 * all) of them to leave the model's normal shipped sampler config
 * completely untouched, exactly as if this mechanism didn't exist. When
 * provided, they're applied as an in-memory, pre-load override (see
 * [com.alibaba.mnnllm.android.llm.SamplerOverrides]) - never written to
 * disk, matching the same stateless-per-call principle as
 * `enable_thinking`. Unlike `enable_thinking` though, these MUST be
 * supplied before the model loads: MNN's Sampler is built once at load
 * time and never re-reads config afterward, so there is no post-load
 * equivalent for these three.
 *
 * The `-p <applicationId>` (or `-n <applicationId>/.benchmark.headless.BenchmarkHeadlessReceiver`
 * for a fully explicit component) is required, not optional: since Android 8,
 * the system refuses to wake a non-running app process to deliver an
 * *implicit* broadcast (action-only, no explicit package/component) to a
 * manifest-registered receiver - it logs
 * "BroadcastQueue: Background execution not allowed: receiving Intent ..."
 * and drops it before onReceive() ever runs. Explicit package/component
 * targeting is exempt from that restriction. Use `.release` as the
 * applicationId suffix for release builds (see app/build.gradle
 * applicationIdSuffix); debug builds use the bare applicationId.
 *
 * Unlike [com.alibaba.mnnllm.api.openai.manager.ApiServiceActionReceiver],
 * this is exported so it's reachable from outside the app at all - that
 * receiver is only ever triggered by in-app PendingIntents while the app
 * process (and its notification) already exists, so it never has to deal
 * with the cold-process delivery restriction above.
 * The actual work happens in [HeadlessBenchmarkService] - this receiver only
 * validates extras and hands off, since BroadcastReceiver.onReceive() has a
 * short execution budget and inference can run for many seconds.
 */
class BenchmarkHeadlessReceiver : BroadcastReceiver() {

    companion object {
        const val ACTION_RUN_PROMPT = "com.mnnllmchat.RUN_PROMPT"
        const val EXTRA_MODEL_PATH = "model_path"
        const val EXTRA_PROMPT = "prompt"
        const val EXTRA_RUN_ID = "run_id"
        const val EXTRA_MAX_TOKENS = "max_tokens"
        const val DEFAULT_MAX_TOKENS = 4096
        const val EXTRA_TOP_K = "top_k"
        const val EXTRA_TOP_P = "top_p"
        const val EXTRA_MIN_P = "min_p"
        const val EXTRA_BACKEND_TYPE = "backend_type"
        private const val TAG = "BenchmarkHeadlessReceiver"
    }

    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action != ACTION_RUN_PROMPT) {
            return
        }
        val modelPath = intent.getStringExtra(EXTRA_MODEL_PATH)
        val prompt = intent.getStringExtra(EXTRA_PROMPT)
        val runId = intent.getStringExtra(EXTRA_RUN_ID)
        val maxTokens = intent.getIntExtra(EXTRA_MAX_TOKENS, DEFAULT_MAX_TOKENS)
        // hasExtra() gates each read so an omitted extra stays genuinely
        // absent (null) rather than falling back to some sentinel value -
        // "not provided" must mean "don't touch the model's default sampler
        // config at all", not "override it with 0".
        val topK = if (intent.hasExtra(EXTRA_TOP_K)) intent.getIntExtra(EXTRA_TOP_K, 0) else null
        val topP = if (intent.hasExtra(EXTRA_TOP_P)) intent.getFloatExtra(EXTRA_TOP_P, 0f) else null
        val minP = if (intent.hasExtra(EXTRA_MIN_P)) intent.getFloatExtra(EXTRA_MIN_P, 0f) else null
        val backendType = intent.getStringExtra(EXTRA_BACKEND_TYPE)
        Log.i(
            TAG,
            "Received RUN_PROMPT run_id=$runId model_path=$modelPath max_tokens=$maxTokens " +
                "top_k=$topK top_p=$topP min_p=$minP backend_type=$backendType"
        )

        if (modelPath.isNullOrBlank() || prompt.isNullOrBlank() || runId.isNullOrBlank()) {
            Log.i(
                HeadlessBenchmarkRunner.LOG_TAG,
                "run_id=${runId ?: "unknown"} RUN_ERROR reason=missing_extras " +
                    "message=model_path, prompt and run_id are all required"
            )
            return
        }

        val serviceIntent = Intent(context, HeadlessBenchmarkService::class.java).apply {
            putExtra(EXTRA_MODEL_PATH, modelPath)
            putExtra(EXTRA_PROMPT, prompt)
            putExtra(EXTRA_RUN_ID, runId)
            putExtra(EXTRA_MAX_TOKENS, maxTokens)
            topK?.let { putExtra(EXTRA_TOP_K, it) }
            topP?.let { putExtra(EXTRA_TOP_P, it) }
            minP?.let { putExtra(EXTRA_MIN_P, it) }
            backendType?.let { putExtra(EXTRA_BACKEND_TYPE, it) }
        }
        // startForegroundService() from a BroadcastReceiver responding to an
        // external broadcast is an explicitly permitted background-start
        // exemption on Android 8+.
        context.startForegroundService(serviceIntent)
    }
}
