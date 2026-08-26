package com.alibaba.mnnllm.android.benchmark.headless

import android.content.Context
import android.os.BatteryManager
import android.util.Log
import java.io.File
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.Executors
import java.util.concurrent.ScheduledExecutorService
import java.util.concurrent.TimeUnit
import kotlin.math.abs

/**
 * Samples battery current during inference and reports the average in mA.
 *
 * Three-tier fallback, ported directly from a sibling project's
 * BenchmarkService.kt (readCurrentUa()/computeAvgCurrentUa() at
 * /Users/ksd/SLM_Factory-SmolChat/SmolChat-Android/app/src/main/java/io/shubham0204/smollmandroid/headless/BenchmarkService.kt)
 * to match its exact behavior rather than reconstructing it:
 *   1. [BatteryManager.BATTERY_PROPERTY_CURRENT_NOW] (microamps) via
 *      getLongProperty() - the normal path, works on most devices.
 *   2. If that returns the "unsupported" sentinel (Long.MIN_VALUE), read
 *      raw current directly from common sysfs paths ([CURRENT_SYSFS_PATHS]).
 *   3. If the WHOLE sampling window produced zero valid instantaneous
 *      readings from either of the above, fall back to a charge-counter
 *      delta: total charge drained ([BatteryManager.BATTERY_PROPERTY_CHARGE_COUNTER])
 *      over the elapsed duration, converted to an equivalent average current.
 */
class PowerSampler(context: Context) {

    private val batteryManager =
        context.applicationContext.getSystemService(Context.BATTERY_SERVICE) as BatteryManager
    private val samplesUa = ConcurrentLinkedQueue<Long>()
    private var executor: ScheduledExecutorService? = null
    private var startTimeMs: Long = 0L
    private var chargeAtStartUah: Long = Long.MIN_VALUE

    fun start(intervalMs: Long = 100) {
        stop()
        startTimeMs = System.currentTimeMillis()
        chargeAtStartUah = try {
            batteryManager.getLongProperty(BatteryManager.BATTERY_PROPERTY_CHARGE_COUNTER)
        } catch (e: Exception) {
            Long.MIN_VALUE
        }
        val exec = Executors.newSingleThreadScheduledExecutor()
        executor = exec
        exec.scheduleAtFixedRate({
            try {
                val sample = readCurrentUa()
                if (sample > 0L) {
                    samplesUa.add(sample)
                }
            } catch (e: Exception) {
                Log.w(TAG, "power sample failed: ${e.message}")
            }
        }, 0, intervalMs, TimeUnit.MILLISECONDS)
    }

    fun stop() {
        executor?.shutdownNow()
        executor = null
    }

    fun reset() {
        stop()
        samplesUa.clear()
        startTimeMs = 0L
        chargeAtStartUah = Long.MIN_VALUE
    }

    /**
     * Average current in mA - from instantaneous samples if any were valid,
     * otherwise the charge-counter-delta fallback over the elapsed duration
     * since [start]. Null if neither produced a usable value.
     */
    fun getAverageMa(): Double? {
        val durationSecs = ((System.currentTimeMillis() - startTimeMs) / 1000L).toInt()
        val avgUa = computeAvgCurrentUa(samplesUa.toList(), chargeAtStartUah, durationSecs)
        return if (avgUa == Long.MIN_VALUE) null else avgUa / 1000.0
    }

    /**
     * BatteryManager first; on the "unsupported" sentinel, falls back to
     * reading raw current directly from sysfs. Returns microamps (always
     * non-negative - magnitude only, since discharge-current sign convention
     * differs across OEMs), or Long.MIN_VALUE if every source failed.
     */
    private fun readCurrentUa(): Long {
        val apiVal = batteryManager.getLongProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW)
        if (apiVal != Long.MIN_VALUE) {
            return abs(apiVal)
        }
        for (path in CURRENT_SYSFS_PATHS) {
            try {
                return abs(File(path).readText().trim().toLong())
            } catch (e: Exception) {
                // try next path
            }
        }
        return Long.MIN_VALUE
    }

    private fun computeAvgCurrentUa(
        samples: List<Long>,
        chargeAtStartUah: Long,
        durationSecs: Int
    ): Long {
        val valid = samples.filter { it > 0L }
        if (valid.isNotEmpty()) {
            return valid.average().toLong()
        }
        if (chargeAtStartUah != Long.MIN_VALUE && durationSecs > 0) {
            val chargeAtEnd = try {
                batteryManager.getLongProperty(BatteryManager.BATTERY_PROPERTY_CHARGE_COUNTER)
            } catch (e: Exception) {
                Long.MIN_VALUE
            }
            if (chargeAtEnd != Long.MIN_VALUE) {
                val deltaUah = chargeAtStartUah - chargeAtEnd
                if (deltaUah > 0) {
                    return deltaUah * 3600L / durationSecs
                }
            }
        }
        return Long.MIN_VALUE
    }

    companion object {
        private const val TAG = "PowerSampler"
        private val CURRENT_SYSFS_PATHS = listOf(
            "/sys/class/power_supply/battery/current_now",
            "/sys/class/power_supply/Battery/current_now",
            "/sys/class/power_supply/bms/current_now",
        )
    }
}
