import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, RepeatedStratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from scipy.stats import randint


def hyperparameter_tune(base_model, n_iter=10, kfold=5):
    # setting parameters to test
    params = {
        'max_depth': randint(3, 8),
        'n_estimators': randint(50, 300),
        'max_features': ['auto', 'sqrt', 'log2', None],
        'class_weight': ['balanced', 'balanced_subsample', None],
        'sampling_strategy': ['majority', 'not majority', 'all', 'auto']
        }
    # using RandomizedSearch to sample parameters
    search = RandomizedSearchCV(base_model, params, n_iter=n_iter, cv=kfold)
    model = search.fit(X, y)
    hyperparameter_output(model)


def hyperparameter_output(model):
    # getting best parameters from the model
    result = model.fit(X_train, y_train)
    best_est_params = model.best_estimator_.get_params()
    best_score = result.best_score_
    print('Best Estimated Parameters: {}'.format(best_est_params))
    print('Best Score: {}'.format(best_score))


def show_values(pc, fmt="%.2f", **kwargs):
    pc.update_scalarmappable()
    ax = pc.axes
    for p, color, value in zip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.all(color[:3] > 0.5):
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)
        ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kwargs)


def cm2inch(*tupl):
    inch = 2.54
    if type(tupl[0]) == tuple:
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)


def heatmap(AUC, title, xlabel, ylabel, xticklabels, yticklabels, figure_width=40, figure_height=20, correct_orientation=False, cmap='RdBu'):
    fig, ax = plt.subplots()
    c = ax.pcolor(AUC, edgecolors='k', linestyle='dashed', linewidths=0.2, alpha=0.7, cmap=cmap)

    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(AUC.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(AUC.shape[1]) + 0.5, minor=False)
    # set tick labels
    ax.set_xticklabels(xticklabels, minor=False)
    ax.set_yticklabels(yticklabels, minor=False)
    # set title and x/y labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # Remove last blank column
    plt.xlim((0, AUC.shape[1]))
    # Turn off all the ticks
    ax = plt.gca()
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    # Add color bar
    plt.colorbar(c)

    # Add text in each cell
    show_values(c)

    # Proper orientation (origin at the top left instead of bottom left)
    if correct_orientation:
        ax.invert_yaxis()
        ax.xaxis.tick_top()

    # resize
    fig = plt.gcf()
    fig.set_size_inches(cm2inch(figure_width, figure_height))


def plot_classification_report(y_tru, y_prd, figsize=(8, 8), ax=None):
    plt.figure(figsize=figsize)

    xticks = ['precision', 'recall', 'f1-score']
    yticks = list(np.unique(y_test))

    rep = np.array(precision_recall_fscore_support(y_tru, y_prd)).T
    avg = np.mean(rep, axis=0)
    avg[-1] = np.sum(rep[:, -1])
    rep = np.insert(rep, rep.shape[0], avg, axis=0)

    sns.heatmap(rep[0:-1, 0:-1],
                annot=True,
                cbar=True,
                xticklabels=xticks,
                yticklabels=yticks,
                ax=ax,
                cmap=sns.diverging_palette(250, 20, as_cmap=True),
                linecolor="white",
                vmin=0,
                vmax=1,
                linewidth=1).set_title("Classification Report - Label 2\nBalanced Random Forest")


def plot_support(y_tru, y_prd, figsize=(8, 8), ax=None):
    plt.figure(figsize=figsize)

    xticks = ['support']
    yticks = list(np.unique(y_tru))

    rep = np.array(precision_recall_fscore_support(y_tru, y_prd)).T
    avg = np.mean(rep, axis=0)
    avg[-1] = np.sum(rep[:, -1])
    rep = np.insert(rep, rep.shape[0], avg, axis=0)

    sns.heatmap(rep[0:-1, -1:],
                annot=True,
                cbar=True,
                xticklabels=xticks,
                yticklabels=yticks,
                ax=ax,
                fmt="",
                cmap=sns.diverging_palette(250, 20, as_cmap=True),
                linecolor="white",
                linewidth=1).set_title("Support - Label 2\nBalanced Random Forest")


def plot_conf_matrix1():
    cf_neu = cf_matrix.astype('float') / cf_matrix.sum(axis=1)[:, np.newaxis]
    conf_matrix = pd.DataFrame(cf_neu,
                               index=['delay', 'ontime'],
                               columns=['delay', 'ontime'])
    print(conf_matrix)

    plt.subplots(figsize=(8, 8))
    heat_map = sns.heatmap(conf_matrix,
                           cmap=sns.diverging_palette(250, 20, as_cmap=True),
                           annot=True,
                           vmax=1,
                           vmin=0,
                           center=0.5,
                           square=False,
                           linewidths=.3,
                           cbar_kws={"shrink": .8})
    heat_map.figure.tight_layout()
    heat_map.figure.subplots_adjust(top=.95, bottom=0.1, left=0.1)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Normalized Confusion Matrix - Label 1\nBalanced Random Forest')
    plt.show()


def plot_conf_matrix2():
    cf_neu = cf_matrix.astype('float') / cf_matrix.sum(axis=1)[:, np.newaxis]
    conf_matrix = pd.DataFrame(cf_neu,
                               index=['delay>15', 'delay>30', 'ontime'],
                               columns=['delay>15', 'delay>30', 'ontime'])
    print(conf_matrix)

    plt.subplots(figsize=(8, 8))
    heat_map = sns.heatmap(conf_matrix,
                           cmap=sns.diverging_palette(250, 20, as_cmap=True),
                           annot=True,
                           vmax=1,
                           vmin=0,
                           center=0.5,
                           square=False,
                           linewidths=.3,
                           cbar_kws={"shrink": .8})
    heat_map.figure.tight_layout()
    heat_map.figure.subplots_adjust(top=.95, bottom=0.1, left=0.1)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Normalized Confusion Matrix - Label 2\nBalanced Random Forest')
    plt.show()


def plot_conf_matrix3():
    cf_neu = cf_matrix.astype('float') / cf_matrix.sum(axis=1)[:, np.newaxis]
    conf_matrix = pd.DataFrame(cf_neu,
                               index=['delay', 'delay>15', 'delay>30', 'delay>60', 'ontime'],
                               columns=['delay', 'delay>15', 'delay>30', 'delay>60', 'ontime'])
    print(conf_matrix)

    plt.subplots(figsize=(8, 8))
    heat_map = sns.heatmap(conf_matrix,
                           cmap=sns.diverging_palette(250, 20, as_cmap=True),
                           annot=True,
                           vmax=1,
                           vmin=0,
                           center=0.5,
                           square=False,
                           linewidths=.3,
                           cbar_kws={"shrink": .8})
    heat_map.figure.tight_layout()
    heat_map.figure.subplots_adjust(top=.95, bottom=0.1, left=0.1)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Normalized Confusion Matrix - Label 3\nBalanced Random Forest')
    plt.show()


def feature_importance():
    importance = pd.DataFrame({'Feature': X_train.columns, 'Importance': np.round(clf.feature_importances_, 3)})
    importance = importance.sort_values('Importance', ascending=True).set_index('Feature')
    print(importance)
    importance.plot.barh().set_title("Feature Importance\nLabel 2\nBalanced Random Forest")
    # importance.figure.subplots_adjust(top=.95, bottom=0.1, left=0.1)


if __name__ == "__main__":
    df = pd.read_csv(r'/Users/McP/PycharmProjects/MP/combine_data_with_labels_new.csv')
    # Data prep
    random_state = 42
    X = df[['HourlyAltimeterSetting', 'HourlyDewPointTemperature', 'HourlyDryBulbTemperature',
            'HourlyPrecipitation', 'HourlyRelativeHumidity', 'HourlySeaLevelPressure',
            'HourlyStationPressure', 'HourlyVisibility', 'HourlyWetBulbTemperature', 'HourlyWindSpeed']]
    y = df['Label2']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=random_state)

    # Implementing the model
    clf = BalancedRandomForestClassifier()# bootstrap=True, ccp_alpha=0.0, class_weight=None,
                                         # criterion='gini', max_depth=5, max_features=None,
                                         # max_leaf_nodes=None, max_samples=None,
                                         # min_impurity_decrease=0.0,
                                         # min_samples_leaf=1, min_samples_split=2,
                                         # min_weight_fraction_leaf=0.0, n_estimators=103, n_jobs=None,
                                         # oob_score=False, random_state=None, replacement=False,
                                         # sampling_strategy='not majority', verbose=0, warm_start=False)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # tuning the parameters in round 1
    # hyperparameter_tune(clf, 15, 5)

    # Evaluating the algorithm
    print('Accuracy:', (accuracy_score(y_test, y_pred) * 100))
    cf_matrix = confusion_matrix(y_test, y_pred)
    class_matrix = classification_report(y_test, y_pred)

    # visuals
    # plot_conf_matrix1()
    plot_conf_matrix2()
    # plot_conf_matrix3()
    plot_classification_report(y_test, y_pred)
    plot_support(y_test, y_pred)
    feature_importance()
