package dev.kosmx.endoftrip

import dev.kosmx.endoftrip.graph.GTFS
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonNames
import java.nio.file.Path
import kotlin.io.path.Path
import kotlin.io.path.writeText

object Routes {

    data class RouteId(
        val name: String,
        val direction: Boolean,
        val vehicleType: Int,
    )

    @Serializable
    data class RouteInfo(
        val name: String,
        val direction: Boolean,
        @SerialName("route_type")
        val routeType: Int,
        val stops: Array<String>,
    ) {
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (javaClass != other?.javaClass) return false

            other as RouteInfo

            if (direction != other.direction) return false
            if (name != other.name) return false
            if (!stops.contentEquals(other.stops)) return false

            return true
        }

        override fun hashCode(): Int {
            var result = direction.hashCode()
            result = 31 * result + name.hashCode()
            result = 31 * result + stops.contentHashCode()
            return result
        }
    }

    private val json = Json { prettyPrint = true }

    @JvmStatic
    fun main(args: Array<String>) {
        val data = GTFS.getRoutePoints()


        val things = mutableMapOf<RouteId, Array<String>>()

        data.asSequence()
            .filter { it.value.name.isNotBlank() }
            .forEach { (tripId, route) ->
                val id = RouteId(route.name, route.direction, route.type)
                val old = things[id]
                if (old == null || old.size < route.stops.size) {
                    things.putIfAbsent(RouteId(route.name, route.direction, route.type), route.stops)
                }
            }

        val routes =
            things.entries.map { (key, value) -> RouteInfo(key.name, key.direction, key.vehicleType, value) }.toList()
        Path("routes.json").writeText(json.encodeToString(routes))

    }
}