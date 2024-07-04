To reproduce this experiment,

1. Run
  ```bash
  python generate_data.py
  ```
  You may have to set `skip_existing=False`, otherwise runs for which data already exists will be skipped.

2. Gather the results
  ```bash
  python gather_data.py
  ```

3. Plot the results
  ```bash
  python plot.py
  ```
