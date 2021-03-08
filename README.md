### NCAA Bracket Builder
##### Tweak your picks to create real brackets to beat your friends

This is a quick dashboard I threw together to interact with my predictions from the [March Machine Learning Mania 2021](https://www.kaggle.com/c/ncaam-march-mania-2021) competition on Kaggle. You will need to drop some of the Kaggle formatted data and your submission file into the inputs folder and then run the dashboard using [Streamlit](https://www.streamlit.io/). To start, put the necessary data in the input folder and rename or edit the file paths/names at the top of thes streamlit code. Finally, type:

```
streamlit run NCAA_BracketBuilder_Sreamlit.py
```
  
Dependencies:
* [Streamlit](https://github.com/streamlit/streamlit)
* [Graphviz](https://pypi.org/project/graphviz/) - (Python implementation only - standalone not needed)
* [Pandas](https://github.com/pandas-dev)
* [Numpy](https://github.com/numpy/numpy)