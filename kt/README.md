# Java/Kotlin project for data preprocessing/stuff

## Download the data from server.
(not real-time updated, but fairly recent)
### Waring: Dataset is 15GB
```shell
pushd data
wget https://kosmx.dev/iPz39eAjSFnifskPLQoTHpMt/bkk_positions.zip
unzip bkk_positions
popd
```


## Run the process script
### Requires Java 21+ JVM 
```shell
./gradlew run data ../output
```
(to build, `gradlew build` can do)

Data processing will take multiple minutes on a **strong** computer with linux.