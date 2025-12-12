package dev.kosmx.endoftrip

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.channels.ReceiveChannel
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import java.io.PrintStream
import java.nio.file.Path
import kotlin.io.path.Path
import kotlin.io.path.createDirectories
import kotlin.io.path.outputStream
import kotlin.io.path.writeText
import kotlin.system.exitProcess

fun saveJob(pOut: Path): suspend (ReceiveChannel<List<EdgeData>>) -> Unit {
    pOut.createDirectories()

    return { channel ->
        var n = 0
        for (batch in channel) {

            val str = StringBuilder()
            str.appendLine(EdgeData.CSV_HEADER)
            for (entry in batch) {
                str.appendLine(entry.toCsvLine())
            }


            pOut.resolve("data-$n.csv").writeText(str.toString())
            n++
        }
    }
}

private const val BUFFER_SIZE = 1024

fun main(args: Array<String>) {
    println("Hello world")

    if (args.size < 2) {
        println("Usage: <this> DATA_IN, DATA_OUT")
        exitProcess(-1)
    }

    val pIn = Path(args[0])
    val pOut = Path(args[1])

    runBlocking {
        val inputChannel = Channel<List<Update>>(BUFFER_SIZE)
        val readJob = launch(Dispatchers.IO) { updateStream(pIn, inputChannel) }

        val outChannel = Channel<List<EdgeData>>(BUFFER_SIZE)
        val processJob = launch(Dispatchers.Default) { DataProcessing.processUpdates(inputChannel, outChannel, pOut) }

        val saveJob = launch(Dispatchers.Default) { saveJob(pOut)(outChannel) }
        println("All jobs started")

        readJob.join()
        println("Finished reading")
        processJob.join()
        println("Finished processing")
        saveJob.join()
        println("Thxbye")
    }
}