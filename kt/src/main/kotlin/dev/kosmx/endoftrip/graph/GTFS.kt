package dev.kosmx.endoftrip.graph

import java.net.HttpURLConnection
import java.net.URI
import java.util.zip.ZipEntry
import java.util.zip.ZipInputStream

object GTFS {

    // this is going to be huge (50MB)
    class GtfsData(
        // lines of trips.txt
        val trips: Sequence<String>,

        // lines of shapes.txt
        val stops: Sequence<String>,

        private val tripsStream: AutoCloseable,
        private val stopsStream: AutoCloseable,
    ) : AutoCloseable {
        override fun close() {
            tripsStream.close()
            stopsStream.close()
        }

        companion object {
            fun of(zip: () -> ZipInputStream): GtfsData {
                val tripsStream = zip()
                val stopsStream = zip()
                val trips = lineSequence(tripsStream, "trips.txt")
                val shapes = lineSequence(stopsStream, "stop_times.txt")


                return GtfsData(trips, shapes, tripsStream, stopsStream)
            }

            private fun find(stream: ZipInputStream, entry: String): ZipEntry {
                while (true) {
                    val e = stream.nextEntry ?: throw NoSuchElementException("zip has no $entry")
                    if (e.name == entry) return e
                }
            }

            private fun lineSequence(zip: ZipInputStream, name: String): Sequence<String> {
                find(zip, name)
                return zip.bufferedReader().lineSequence()
            }
        }
    }

    private fun load(): ByteArray {
        val connection =
            URI("https://go.bkk.hu/api/static/v1/public-gtfs/budapest_gtfs.zip").toURL()
                .openConnection() as HttpURLConnection
        connection.setRequestMethod("GET")
        connection.getInputStream().use { input ->
            return input.readAllBytes()
        }
    }

    public fun getRoutePoints(): Map<String, Route> {
        val data = load()

        println("GTFS downloaded, start processing...")

        GtfsData.of { ZipInputStream(data.inputStream()) }.use { gtfs ->
            val trips: Map<String, RouteEntry> = gtfs.trips.run {
                val it = iterator()
                val head = it.next().split(",")
                val routeIndex = head.indexOf("route_id")
                val tripIndex = head.indexOf("trip_id")
                val directionIndex = head.indexOf("direction_id")

                it.asSequence().map { it.split(",") }.associate {
                    val route = it[routeIndex]
                    val trip = it[tripIndex]
                    val direction = it[directionIndex].toIntOrNull() == 1

                    trip to RouteEntry(route, direction)
                }
            }

            println("Trips loaded")

            gtfs.stops.run {
                val it = iterator()
                val head = it.next().split(",")
                val tripIndex = head.indexOf("trip_id")
                val stopIndex = head.indexOf("stop_id")
                val sequenceIndex = head.indexOf("stop_sequence")

                it.asSequence().map {
                    it.split(",")
                }.map {
                    it[tripIndex] to StopPosition(it[sequenceIndex].toInt(), it[stopIndex])
                }.forEach { (tripId, stop) ->
                    trips[tripId]?.stops?.add(stop)
                }
            }

            println("Timetables loaded")

            return trips.entries.mapNotNull { (trip, route) ->
                if (route.stops.size < 4) return@mapNotNull null
                if (route.stops.windowed(2).any { (a, b) -> a.index > b.index }) {
                    route.stops.sortBy { it.index }
                }

                trip to Route(route.direction, route.stops.map { it.stop }.toTypedArray())
            }.associate { it }.also {
                println("GTFS graph ready")
            }
        }
    }

    data class RouteEntry(
        val routeId: String,
        val direction: Boolean,
        val stops: MutableList<StopPosition> = mutableListOf()
    )

    data class StopPosition(
        val index: Int,
        val stop: String,
    )
}