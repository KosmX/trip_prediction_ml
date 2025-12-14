plugins {
    // Apply the org.jetbrains.kotlin.jvm Plugin to add support for Kotlin.
    alias(libs.plugins.kotlin.jvm)
    alias(libs.plugins.kotlin.serialization)
    id("com.google.protobuf") version "0.9.5"
    application
}

sourceSets {
    main {
        proto {
            srcDir("src/main/protobuf")
        }
    }
}

application {
    mainClass = "dev.kosmx.endoftrip.ProtoProcessorKt"
}

repositories {
    // Use Maven Central for resolving dependencies.
    mavenCentral()
}

dependencies {

    // https://mvnrepository.com/artifact/com.google.protobuf/protobuf-kotlin
    implementation("com.google.protobuf:protobuf-kotlin:4.33.2")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.10.2")
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.9.0")

    // This dependency is exported to consumers, that is to say found on their compile classpath.
    //api(libs.commons.math3)

    // This dependency is used internally, and not exposed to consumers on their own compile classpath.
    //implementation(libs.guava)
}

kotlin {
    jvmToolchain(21)
}

java {
    javaToolchains {
        version = 21
    }
}
