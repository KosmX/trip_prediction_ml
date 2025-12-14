# BKK ETA estimation with machine learning

## Data preparation
For data preparation, see [kt/README.md](kt/README.md)  
Original data is 15 GB of protobuf files, 4.2 GB compressed.  

Processed data is available at https://kosmx.dev/iPz39eAjSFnifskPLQoTHpMt/bkk.zip   
And will be available at least until 2026-02-01 (but probably much longer).

## Data noisiness..
Soo, because i kinda messed up the data collection (collected vehicle positions), the data precision is up to a minute.  
I do not think it will give problematic result on whole lines, as the average is good, just noisy data.  
However RMSE (MAE) will be a bit more noisy than optimal.

## Run
```shell
docker compose -f compose-nv.yml up --build
```
or on AMD
```shell
docker compose -f compose-amd.yml up --build
```

## API :)
```
wget http://localhost:8080/timetable?route=56
```

I've wasted way more time than should on fixing python types...

## Goal
Instead of trying to predict delay, (which can be problematic, as official timetables already account for some delay), I was trying to estimate the original timetables.  
I really cannot work without types, nothing works, i have no clue why, strings should be strings, int should be insts, etc... 
For, somehow I managed to make it work anyway.

The API appears to have problems, I do not know why, that's not the model what's broken, but somehow the python logic.
