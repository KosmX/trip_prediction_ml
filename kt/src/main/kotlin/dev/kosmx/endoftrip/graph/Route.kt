package dev.kosmx.endoftrip.graph

data class Route(
    val direction: Boolean,
    val stops: Array<String>,
    val name: String,
    val type: Int
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as Route

        if (direction != other.direction) return false
        if (!stops.contentEquals(other.stops)) return false

        return true
    }

    override fun hashCode(): Int {
        var result = direction.hashCode()
        result = 31 * result + stops.contentHashCode()
        return result
    }

    override fun toString(): String {
        return "Route(direction=$direction, stops=${stops.contentToString()})"
    }


}
