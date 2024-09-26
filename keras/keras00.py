import matplotlib.font_manager as fm
path = '/Library/Fonts/NanumBarunpenRegular.otf'
fontprop = fm.FontProperties(fname=path, size=18)
plt.title('비용', fontproperties=fontprop)