@file:OptIn(ExperimentalTime::class)

package dev.kosmx.endoftrip

import kotlin.time.Duration
import kotlin.time.ExperimentalTime
import kotlin.time.Instant

data class EdgeData constructor(
    val from: String,
    val to: String,
    val duration: Duration,
    val vehicleModel: Int?,
    val vehicleType: Int?,
    val time: Instant,
) {
    companion object {
        const val CSV_HEADER = "from,to,duration,vehicle_model,vehicle_type"
    }

    fun toCsvLine() = "$from,$to,${duration.inWholeSeconds},$vehicleModel,$vehicleType,$time"
}
