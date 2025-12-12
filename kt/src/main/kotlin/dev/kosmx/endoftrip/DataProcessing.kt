@file:OptIn(ExperimentalTime::class)

package dev.kosmx.endoftrip

import dev.kosmx.endoftrip.graph.GTFS
import dev.kosmx.endoftrip.trip.Trip
import io.realcity.datamodel.gtfs_rt.GtfsRealtime
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.channels.ReceiveChannel
import kotlinx.coroutines.channels.SendChannel
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.chunked
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.yield
import realcity.GtfsRealtimeRealcity
import java.nio.file.Path
import kotlin.io.path.writeText
import kotlin.math.pow
import kotlin.time.Duration.Companion.hours
import kotlin.time.ExperimentalTime
import kotlin.time.Instant

object DataProcessing {
    /**
     * Consumes data from inChannel, send data to outChannel in batches
     */
    @OptIn(ExperimentalCoroutinesApi::class)
    suspend fun processUpdates(
        inChannel: ReceiveChannel<List<Update>>,
        outChannel: SendChannel<List<EdgeData>>,
        pOut: Path
    ) =
        coroutineScope {
            val input = flow {
                for (updates in inChannel) {

                    for (update in updates) {
                        emit(update)
                    }
                }
            }


            processor(input, pOut).chunked(1 shl 20).collect {
                outChannel.send(it)
            }
            outChannel.close()

        }

    private fun processor(input: Flow<Update>, pOut: Path): Flow<EdgeData> = flow {
        // expensive operation :D
        val bkkData = GTFS.getRoutePoints()

        val tripCache = bkkData.asSequence().associate { (trip, route) ->
            trip to Trip(route.stops[0], Instant.DISTANT_PAST)
        }.toMutableMap()
        println("building trips complete")

        /**
        bkkData.entries.first().let { (id, route) ->
        println("$id: $route")
        }
         */

        val models = mutableMapOf<String, Int>()

        input.collect { (_, update) ->
            // if vehicle is not in trip, ignore
            if (!update.hasTrip() || !update.trip.hasTripId()) return@collect

            val trip = update.trip
            val timestamp = Instant.fromEpochSeconds(update.timestamp)
            val tracker = tripCache[trip.tripId] ?: return@collect
            val stops = bkkData[trip.tripId]!! // will never be null if tracker is not null

            // don't increase time if vehicle is waiting at start
            if (tracker.lastStop == update.stopId && tracker.lastStop == stops.stops[0]) {
                tripCache[trip.tripId] = tracker.copy(lastUpdate = timestamp)
                return@collect // nothing to report
            }

            val start = stops.stops.indexOf(tracker.lastStop)
            val stop = stops.stops.indexOf(update.stopId)
            if (start < 0 || stop < 0){
                return@collect
            } // wat???

            val duration = timestamp - tracker.lastUpdate
            if (start + 5 >= stop && duration < 1.hours) {
                if (start < stop) { // advanced at least one stop
                    // safety, teleportation **does not** exists
                    val increment = duration / (stop - start)

                    val extension = update.vehicle.getExtension(GtfsRealtimeRealcity.vehicle)
                    val model = if (extension.hasVehicleModel()) extension.vehicleModel else null
                    val modelId = model?.let { models.getOrPut(it) { models.size } }


                    for (i in start..<stop) {
                        val from = stops.stops[i]
                        val to = stops.stops[i + 1]

                        emit(
                            EdgeData(
                                from,
                                to,
                                increment,
                                modelId,
                                if (extension.hasVehicleType()) extension.vehicleType else null,
                                timestamp
                            )
                        )
                    }
                    tripCache[trip.tripId] = tracker.copy(lastUpdate = timestamp, lastStop = update.stopId)
                }
            } else {
                tripCache[trip.tripId] = tracker.copy(lastUpdate = timestamp, lastStop = update.stopId)
            }
        }
        println("Processing last update is complete")

        pOut.resolve("vehicles.txt").writeText(
            models.entries.sortedBy { it.value }.joinToString("\n") { it.key }
        )
    }
}