package com.alibaba.mnnllm.android.benchmark.headless

import android.content.Context
import android.os.Build
import android.os.PowerManager
import android.util.Log
import java.io.File
import java.util.concurrent.Executors
import java.util.concurrent.ScheduledExecutorService
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference

/**
 * Tracks device thermal state during a benchmark run:
 *  - [PowerManager.getCurrentThermalStatus] for the coarse OS-level status.
 *  - Real hardware sensor temperatures read directly from
 *    /sys/class/thermal/thermal_zoneN/temp, sampled roughly once per second,
 *    reporting the peak seen for the CPU zone and the skin/skin-adjacent zone.
 *
 * Zone numbering is NOT stable across devices, and is not guaranteed stable
 * across reboots on the same device, so zones are always matched by their
 * "type" string (e.g. containing "cpu" or "skin") rather than by index.
 */
class ThermalSampler(private val context: Context) {

    private val cpuZoneTempFile: File?
    private val skinZoneTempFile: File?

    private val maxCpuTempC = AtomicReference<Double?>(null)
    private val maxSkinTempC = AtomicReference<Double?>(null)
    private var executor: ScheduledExecutorService? = null

    init {
        val zones = discoverZones()
        cpuZoneTempFile = zones.firstOrNull { it.first.contains("cpu") }?.second
        skinZoneTempFile = zones.firstOrNull { it.first.contains("skin") }?.second
        Log.d(
            TAG,
            "discovered ${zones.size} thermal zones; cpu=${cpuZoneTempFile?.path ?: "not found"}, " +
                "skin=${skinZoneTempFile?.path ?: "not found"}"
        )
    }

    private fun discoverZones(): List<Pair<String, File>> {
        val result = mutableListOf<Pair<String, File>>()
        var index = 0
        while (index < MAX_ZONES) {
            val zoneDir = File("/sys/class/thermal/thermal_zone$index")
            if (!zoneDir.exists()) {
                break
            }
            try {
                val typeFile = File(zoneDir, "type")
                val tempFile = File(zoneDir, "temp")
                if (typeFile.canRead() && tempFile.exists()) {
                    val type = typeFile.readText().trim().lowercase()
                    result.add(type to tempFile)
                }
            } catch (e: Exception) {
                Log.w(TAG, "failed to read thermal zone $index: ${e.message}")
            }
            index++
        }
        return result
    }

    private fun readZoneTempC(file: File?): Double? {
        if (file == null) return null
        return try {
            // Values under thermal_zoneN/temp are in millidegrees Celsius.
            file.readText().trim().toLong() / 1000.0
        } catch (e: Exception) {
            null
        }
    }

    fun start(intervalMs: Long = 1000) {
        stop()
        val exec = Executors.newSingleThreadScheduledExecutor()
        executor = exec
        exec.scheduleAtFixedRate({
            readZoneTempC(cpuZoneTempFile)?.let { temp ->
                maxCpuTempC.updateAndGet { current -> if (current == null || temp > current) temp else current }
            }
            readZoneTempC(skinZoneTempFile)?.let { temp ->
                maxSkinTempC.updateAndGet { current -> if (current == null || temp > current) temp else current }
            }
        }, 0, intervalMs, TimeUnit.MILLISECONDS)
    }

    fun stop() {
        executor?.shutdownNow()
        executor = null
    }

    fun reset() {
        stop()
        maxCpuTempC.set(null)
        maxSkinTempC.set(null)
    }

    /** Peak CPU zone temperature seen since [start], in Celsius, or null if that zone was never found/readable. */
    fun getMaxCpuTempC(): Double? = maxCpuTempC.get()

    /** Peak skin zone temperature seen since [start], in Celsius, or null if that zone was never found/readable. */
    fun getMaxSkinTempC(): Double? = maxSkinTempC.get()

    /** Coarse OS thermal-throttling status label. "unavailable" below API 29 or on read failure. */
    fun getThermalStatusLabel(): String {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) {
            return "unavailable"
        }
        return try {
            val powerManager = context.applicationContext.getSystemService(Context.POWER_SERVICE) as PowerManager
            statusLabel(powerManager.currentThermalStatus)
        } catch (e: Exception) {
            "unavailable"
        }
    }

    companion object {
        private const val TAG = "ThermalSampler"
        private const val MAX_ZONES = 60

        private fun statusLabel(status: Int): String = when (status) {
            PowerManager.THERMAL_STATUS_NONE -> "Normal"
            PowerManager.THERMAL_STATUS_LIGHT -> "Light"
            PowerManager.THERMAL_STATUS_MODERATE -> "Moderate"
            PowerManager.THERMAL_STATUS_SEVERE -> "Severe"
            PowerManager.THERMAL_STATUS_CRITICAL -> "Critical"
            PowerManager.THERMAL_STATUS_EMERGENCY -> "Emergency"
            PowerManager.THERMAL_STATUS_SHUTDOWN -> "Shutdown"
            else -> "Unknown($status)"
        }
    }
}
