@file:OptIn(ExperimentalTime::class)

package dev.kosmx.endoftrip

import com.google.protobuf.ExtensionRegistry
import io.realcity.datamodel.gtfs_rt.GtfsRealtime
import kotlinx.coroutines.channels.SendChannel
import realcity.GtfsRealtimeRealcity
import java.nio.file.Path
import kotlin.coroutines.cancellation.CancellationException
import kotlin.io.path.inputStream
import kotlin.io.path.listDirectoryEntries
import kotlin.io.path.name
import kotlin.time.ExperimentalTime
import kotlin.time.Instant

data class Update(
    val messageId: String,
    val update: GtfsRealtime.VehiclePosition
)

private data class ArchiveEntry(
    val file: Path,
    val timestamp: Instant
)

private val pattern = Regex("(?<name>\\w+) (?<date>[^\\n.]+).pb")

private val extensionRegistry = ExtensionRegistry.newInstance().apply {
    add(GtfsRealtimeRealcity.vehicle)
}

suspend fun updateStream(path: Path, outChannel: SendChannel<List<Update>>) {
    val entries = path.listDirectoryEntries("*.pb")
        .map { entry ->
            val match = pattern.matchEntire(entry.name)
            require(match != null) { "Failed to match filename: ${entry.name}" }
            val instantStr = match.groups["date"]!!.value

            // Attempt this
            ArchiveEntry(entry, Instant.parse(instantStr.replace(" ", "T")))
        }.sortedBy { it.timestamp }

    println("Having ${entries.size} entries.")

    for (entry in entries) {
        val updates = try {
            val entries = entry.file.inputStream().buffered().use { inputStream ->
                GtfsRealtime.FeedMessage.parseFrom(inputStream, extensionRegistry)
            }
            entries.entityList
                .filter {
                    it.hasVehicle()
                }.map {
                    Update(it.id, it.vehicle)
                }.toList()
        } catch (e: Throwable) {
            println("Error while reading protobuf: ${entry.file}")
            e.printStackTrace()
            if (e is CancellationException) throw e else null
        }
        if (updates != null) {
            outChannel.send(updates)
        }
    }
    outChannel.close()
}