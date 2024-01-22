#!/bin/bash

# Copyright 2024 Iman Samizadeh
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Contact information:
# Iman Samizadeh
# Email: iman.samizadeh@gmail.com
## https://www.dukascopy-node.app/config/cli

instrument="btcusd"
#granularity="m1"
granularity="d1"
output_format="csv"

date_ranges=("2017-02-01" "2018-02-01" "2019-02-01" "2020-02-01" "2021-02-01" "2022-02-01" "2023-02-01")
to_dates=("2017-12-31" "2018-12-31" "2019-12-31" "2020-12-31" "2021-12-31" "2022-12-31" "2023-12-31")

output_directory="./data/$instrument"

for i in "${!date_ranges[@]}"; do
  from_date="${date_ranges[$i]}"
  to_date="${to_dates[$i]}"
  npx dukascopy-cli -i $instrument -from $from_date -to $to_date -t $granularity -f $output_format -bs 15 -bp 2000 --volumes true --directory $output_directory
  echo "Data download completed for date range: $from_date to $to_date"
done

array_length=${#to_dates[@]}
last_index=$((array_length - 1))
last_to_date=${to_dates[$last_index]}
current_date=$(date +%Y-%m-%d)

npx dukascopy-cli -i $instrument -from $last_to_date -to $current_date -t $granularity -f $output_format -bs 15 -bp 2000 --volumes true --directory $output_directory
echo "Latest data download completed for date range: $last_to_date to $current_date"
