@file:OptIn(ExperimentalTime::class)

package dev.kosmx.endoftrip.trip

import kotlin.time.ExperimentalTime
import kotlin.time.Instant

data class Trip(
    var lastStop: String,
    var lastUpdate: Instant
)
