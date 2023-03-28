# Super Resolution

## How to run the super resolution script

Place the script above the train/test/val folder (or create a symbolic link to those folders in the same directory as the super resolution script).

Make sure the folders are named correctly :
- **train** for the train dataset
- **test** for the test dataset
- **val** for the validation dataset
- **output/super_res_model** where the super resolution model should be placed
- **output/visualizations** where the output will be place

## Output

The output images are going to be inside a folder with the method name
- **output/visualizations/bicubic/** for bicubic method
- **output/visualizations/super-res/** for super-res method

### Metrics

Use **../statistic/statisc_rain100_super_res.m** and **../statistic/statisc_rain100_super_res.m** and make sure **PReNet** variable points to the right path.