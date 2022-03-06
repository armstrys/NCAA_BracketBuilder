### NCAA Bracket Builder
##### Sense check your model and check your results by simulating real brackets

I am not currently hosting a dashboard for this because I don't want to make the data publicly availilbe through Githib per the Kaggle rules around redistributing data. If you have an idea of how this could be easily replicated through the Kaggle notebook interface, please let me know!

This is a dashboard I threw together to interact with my predictions from the [March Machine Learning Mania 2022 - Men's](https://www.kaggle.com/c/mens-march-mania-2022) and [March Machine Learning Mania 2022 - Women's](https://www.kaggle.com/c/mens-march-mania-2022) competitions on Kaggle. You will need to drop some of the Kaggle formatted data and your submission file into the inputs folder and then run the dashboard using [Streamlit](https://www.streamlit.io/). To start, put the necessary data in the input folder and rename or edit the file paths/names at the top of the Streamlit code. Finally, type:

```
streamlit run NCAA_BracketBuilder_Sreamlit.py
```
  
Dependencies:
* [Streamlit](https://github.com/streamlit/streamlit)
* [Graphviz](https://pypi.org/project/graphviz/) - (Python implementation only - standalone not needed)
* [Pandas](https://github.com/pandas-dev)
* [Numpy](https://github.com/numpy/numpy)