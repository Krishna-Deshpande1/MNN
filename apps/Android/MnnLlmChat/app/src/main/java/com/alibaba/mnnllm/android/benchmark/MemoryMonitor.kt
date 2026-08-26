package com.alibaba.mnnllm.android.benchmark

import android.util.Log
import java.io.File
import java.util.Timer
import java.util.TimerTask
import java.util.concurrent.ConcurrentLinkedQueue

object MemoryMonitor {

    private var maxPssKb: Long = 0
    private val memoryHistory = ConcurrentLinkedQueue<Long>()
    private var timer: Timer? = null

    fun start(intervalSeconds: Long = 5) {
        stop() // Stop any existing timer

        // Take one synchronous, blocking sample right now, on the caller's
        // thread, before the Timer even exists. A java.util.Timer's first
        // tick is not truly immediate even with delay=0: Timer() spins up a
        // new background thread, and under the CPU contention of model
        // loading starting right after this call, that thread's first
        // actual run() can be scheduled late enough that a short request
        // (e.g. ~700ms end to end) finishes before the timer fires even
        // once - leaving maxPssKb at its stale default. This guarantees at
        // least one real reading exists no matter how fast the request
        // completes.
        sampleOnce()

        timer = Timer()
        timer?.schedule(object : TimerTask() {
            override fun run() {
                sampleOnce()
            }
        }, 0, intervalSeconds * 1000)
    }

    /**
     * Forces one additional synchronous sample right now, on the caller's
     * thread. Use this at a specific lifecycle point the periodic timer
     * might not reach in time for a fast caller - e.g. right after model
     * load completes, since the start()-time sample necessarily fires
     * before load and only captures pre-load baseline memory.
     */
    fun sampleNow() {
        sampleOnce()
    }

    private fun sampleOnce() {
        val currentPssKb = readVmRssKb()
        if (currentPssKb < 0) {
            return
        }

        // Guaranteed initial sample (caller thread) and periodic samples
        // (timer thread) can now race on maxPssKb, so serialize updates.
        var maxAfterUpdate: Long
        synchronized(this) {
            if (currentPssKb > maxPssKb) {
                maxPssKb = currentPssKb
            }
            maxAfterUpdate = maxPssKb
        }
        memoryHistory.add(currentPssKb)

        Log.d("MemoryMonitor", "Current PSS: ${currentPssKb}KB, Max PSS: ${maxAfterUpdate}KB")
    }

    /**
     * Reads resident set size directly from the kernel via
     * /proc/self/status's "VmRSS:" line (format: "VmRSS:    627432 kB"),
     * rather than Debug.getMemoryInfo()/totalPss. The latter goes through
     * Android's PSS collection service, which was found to return the same
     * cached value across many consecutive calls within a short window - a
     * documented throttling characteristic of that API on-device.
     * /proc/self/status is a raw kernel file read with no such caching
     * layer, so each read reflects genuinely current memory state.
     */
    private fun readVmRssKb(): Long {
        return try {
            File("/proc/self/status").useLines { lines ->
                lines.firstOrNull { it.startsWith("VmRSS:") }
                    ?.trim()
                    ?.split(Regex("\\s+"))
                    ?.getOrNull(1)
                    ?.toLongOrNull()
            } ?: -1L
        } catch (e: Exception) {
            Log.w("MemoryMonitor", "failed to read /proc/self/status: ${e.message}")
            -1L
        }
    }

    fun stop() {
        timer?.cancel()
        timer = null
    }

    fun getMaxMemoryPssKb(): Long {
        synchronized(this) {
            return maxPssKb
        }
    }

    fun getMemoryHistory(): List<Long> {
        return memoryHistory.toList()
    }

    fun reset() {
        stop()
        synchronized(this) {
            maxPssKb = 0
        }
        memoryHistory.clear()
    }
}