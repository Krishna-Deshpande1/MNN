package com.alibaba.mnnllm.android.benchmark.headless

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import android.util.Log
import androidx.core.app.NotificationCompat
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch

/**
 * Foreground service that owns the lifecycle of a single headless benchmark
 * run, modeled after [com.alibaba.mnnllm.api.openai.service.OpenAIService].
 *
 * Only ever started by [BenchmarkHeadlessReceiver]. Each onStartCommand()
 * runs exactly one isolated request via [HeadlessBenchmarkRunner] and then
 * stops itself (START_NOT_STICKY) - there is no notion of a long-lived
 * "server" here, unlike OpenAIService.
 */
class HeadlessBenchmarkService : Service() {

    private val serviceScope = CoroutineScope(Dispatchers.IO + SupervisorJob())

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val runId = intent?.getStringExtra(BenchmarkHeadlessReceiver.EXTRA_RUN_ID) ?: "unknown"
        val modelPath = intent?.getStringExtra(BenchmarkHeadlessReceiver.EXTRA_MODEL_PATH)
        val prompt = intent?.getStringExtra(BenchmarkHeadlessReceiver.EXTRA_PROMPT)
        val maxTokens = intent?.getIntExtra(BenchmarkHeadlessReceiver.EXTRA_MAX_TOKENS, BenchmarkHeadlessReceiver.DEFAULT_MAX_TOKENS)
            ?: BenchmarkHeadlessReceiver.DEFAULT_MAX_TOKENS
        val topK = if (intent?.hasExtra(BenchmarkHeadlessReceiver.EXTRA_TOP_K) == true) {
            intent.getIntExtra(BenchmarkHeadlessReceiver.EXTRA_TOP_K, 0)
        } else null
        val topP = if (intent?.hasExtra(BenchmarkHeadlessReceiver.EXTRA_TOP_P) == true) {
            intent.getFloatExtra(BenchmarkHeadlessReceiver.EXTRA_TOP_P, 0f)
        } else null
        val minP = if (intent?.hasExtra(BenchmarkHeadlessReceiver.EXTRA_MIN_P) == true) {
            intent.getFloatExtra(BenchmarkHeadlessReceiver.EXTRA_MIN_P, 0f)
        } else null

        val notification = buildNotification(runId)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(NOTIFICATION_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC)
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }

        if (modelPath.isNullOrBlank() || prompt.isNullOrBlank()) {
            Log.i(
                HeadlessBenchmarkRunner.LOG_TAG,
                "run_id=$runId RUN_ERROR reason=missing_extras message=model_path and prompt are required"
            )
            finishAndStop(startId)
            return START_NOT_STICKY
        }

        serviceScope.launch {
            try {
                HeadlessBenchmarkRunner(applicationContext).run(modelPath, prompt, runId, maxTokens, topK, topP, minP)
            } finally {
                finishAndStop(startId)
            }
        }
        return START_NOT_STICKY
    }

    private fun finishAndStop(startId: Int) {
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                stopForeground(STOP_FOREGROUND_REMOVE)
            } else {
                @Suppress("DEPRECATION")
                stopForeground(true)
            }
        } catch (e: Exception) {
            Log.w(TAG, "stopForeground failed: ${e.message}")
        }
        stopSelf(startId)
    }

    override fun onDestroy() {
        serviceScope.cancel()
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    private fun createNotificationChannel() {
        val manager = getSystemService(NotificationManager::class.java) ?: return
        val channel = NotificationChannel(
            CHANNEL_ID,
            "Headless Benchmark",
            NotificationManager.IMPORTANCE_LOW
        )
        manager.createNotificationChannel(channel)
    }

    private fun buildNotification(runId: String): Notification {
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("Running headless benchmark")
            .setContentText("run_id=$runId")
            .setSmallIcon(android.R.drawable.ic_dialog_info)
            .setOngoing(true)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .build()
    }

    companion object {
        private const val TAG = "HeadlessBenchmarkService"
        private const val CHANNEL_ID = "headless_benchmark_channel"
        private const val NOTIFICATION_ID = 2001
    }
}
