from utils import *
from prophets import *

"""
Question 1.
2 Prophets 1 Game.
You may change the input & output parameters of the function as you wish.
Assuming that the dataset is named 'gt' with shape (100, 1000)
The first axis represents train sets, and the second axis represents 
game results
"""

EXP_NUM = 100
PROPHET_NUM = 500
EPSILON = 0.01


def Scenario_1(train_set, test_set, size):
    p1 = Prophet(0.2)
    p2 = Prophet(0.4)
    test_err, est_err, p1_wins = 0, 0, 0
    for i in range(EXP_NUM):
        train_set_reduced = np.random.choice(train_set[i, :], size=size)
        # Create Prophet's prediction vectors
        p1_predictions = p1.predict(train_set_reduced)
        p2_predictions = p2.predict(train_set_reduced)
        # Create error estimations for this round
        p1_err = compute_error(p1_predictions, train_set_reduced)
        p2_err = compute_error(p2_predictions, train_set_reduced)
        # Choose winning prophet
        random_value = np.random.choice([0, 1])
        if (p1_err < p2_err) or (p1_err == p2_err and random_value == 0):
            p1_wins += 1
            test_err += compute_error(p1.predict(test_set), test_set)
        elif (p1_err > p2_err) or (p1_err == p2_err and random_value == 1):
            test_err += compute_error(p2.predict(test_set), test_set)
            est_err += 0.2
    print("Number of times best prophet selected:", p1_wins)
    print("Average test error of selected prophet:", test_err / EXP_NUM)
    print("Average approximation error: 0.2")
    print("Average estimation error:", est_err / EXP_NUM, "\n")


def Scenario_2(train_set, test_set, size):
    """
    Question 2.
    2 Prophets 10 Games.
    You may change the input & output parameters of the function as you wish.
    """
    Scenario_1(train_set, test_set, size)


def Scenario_3(train_set, test_set, k, m, min_p, max_p):
    """
    Question 3.
    500 Prophets 10 Games.
    You may change the input & output parameters of the function as you wish.
    """
    # prophet creation and finding the best prophet
    prophets = sample_prophets(k, min_p, max_p)
    true_err_rates = np.array([prophet.err_prob for prophet in prophets])
    min_index = np.argmin(true_err_rates)
    best_err_rate = prophets[min_index].err_prob
    est_err, test_err, true_err_rates, best_wins, almost_best_wins = 0, 0, 0, 0, 0
    train_gen_err, test_gen_err = 0, 0

    for exp in range(EXP_NUM):
        train_set_reduced = np.random.choice(train_set[exp, :], size=m)
        predictions = np.zeros((k, m)) # Initialize predict array of 500 X 10
        empiric_error_rates = np.zeros(k) # Initialize empiric_error_rates array[500]
        for index in range(k):
            # make prediction for current prophet
            predictions[index, :] = prophets[index].predict(train_set_reduced)
            empiric_error_rates[index] = compute_error(predictions[index], train_set_reduced)
        # find best prophet
        cur_best_index = np.argmin(empiric_error_rates)
        cur_best_err = np.min(empiric_error_rates)
        # check if the current best prophet is the actual best prophet
        if cur_best_index == min_index:
            best_wins += 1
            almost_best_wins += 1
        else:
            est_err += abs(best_err_rate - prophets[cur_best_index].err_prob)
            if abs(best_err_rate - prophets[cur_best_index].err_prob) < EPSILON:
                almost_best_wins += 1
        # compute errors
        cur_test_pred = prophets[cur_best_index].predict(test_set)
        test_err += compute_error(cur_test_pred, test_set)
        train_gen_err += abs(cur_best_err - prophets[cur_best_index].err_prob)
        test_gen_err += abs(cur_best_err - compute_error(cur_test_pred, test_set))
    print("Average test error of selected prophet:", test_err / EXP_NUM)
    print("Number of times best prophet selected:", best_wins)
    print("Number of times prophet selected within epsilon:", almost_best_wins)
    print("Average approximation error:", best_err_rate)
    print("Average estimation error:", est_err / EXP_NUM)
    print("Average train generalization error:", train_gen_err / EXP_NUM)
    print("Average test generalization error:", test_gen_err / EXP_NUM, "\n")
    return best_err_rate, est_err / EXP_NUM


def Scenario_4(train_set, test_set, k, m, min_p, max_p):
    """
    Question 4.
    500 Prophets 1000 Games.
    You may change the input & output parameters of the function as you wish.
    """
    ############### YOUR CODE GOES HERE ###############
    Scenario_3(train_set, test_set, k, m, min_p, max_p)


def Scenario_5(train_set, test_set):
    """
    Question 5.
    School of Prophets.
    You may change the input & output parameters of the function as you wish.
    """
    # error tables initialization
    approx_err_table = np.zeros((4, 4))
    est_err_table = np.zeros((4, 4))
    test_err_table = np.zeros((4, 4))
    i, j = -1, -1  # indexes
    for k in [2, 5, 10, 50]:
        i += 1
        j = -1
        prophets = sample_prophets(k, 0, 0.2)
        true_risks = np.array([prophet.err_prob for prophet in prophets])
        opt_err = np.min(true_risks)
        for m in [1, 10, 50, 1000]:
            j += 1
            est_err, test_err = 0, 0
            for exp in range(EXP_NUM):
                train_set_reduced = np.random.choice(train_set[exp, :], size=m)
                predictions = np.zeros((k, m))  # Initialize predict array of 500 X 10
                empiric_errors = np.zeros(k)  # Initialize empiric_errors array[500]
                for index in range(k):
                    # make prediction for current prophet
                    predictions[index, :] = prophets[index].predict(train_set_reduced)
                    empiric_errors[index] = compute_error(predictions[index], train_set_reduced)
                # calculate errors
                est_err += abs(opt_err - true_risks[np.argmin(empiric_errors)])
                test_pred = prophets[np.argmin(empiric_errors)].predict(test_set)
                test_err += compute_error(test_pred, test_set)
            approx_err_table[i][j] = opt_err
            est_err_table[i][j] = est_err / EXP_NUM
            test_err_table[i][j] = test_err / EXP_NUM
            print("number of prophets:", k)
            print("number of games in train set:", m)
            print("Average test error of selected prophet:", test_err / EXP_NUM)
            print("Average approximation error:", opt_err)
            print("Average estimation error:", est_err / EXP_NUM)
    Scenario5_table(test_err_table, approx_err_table, est_err_table)
    Heatmap(approx_err_table, est_err_table)


def Heatmap(approx_err_table, est_err_table):
    columns = ['M = 1', 'M = 10', 'M = 50', 'M = 1000']
    rows = ['K = 2', 'K = 5', 'K = 10', 'K = 50']
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # Plot the first heatmap
    im1 = axes[0].imshow(approx_err_table, cmap='Blues', interpolation='nearest', aspect='auto')
    axes[0].set_title('Average Approximation Error')
    axes[0].set_xticks(np.arange(len(columns)))
    axes[0].set_yticks(np.arange(len(rows)))
    axes[0].set_xticklabels(columns)
    axes[0].set_yticklabels(rows)
    for i in range(len(rows)):
        for j in range(len(columns)):
            axes[0].text(j, i, f'{approx_err_table[i, j]:.4f}', ha='center', va='center', color='black')
    # Plot the second heatmap
    im2 = axes[1].imshow(est_err_table, cmap='Blues', interpolation='nearest', aspect='auto')
    axes[1].set_title('Average Estimation Error')
    axes[1].set_xticks(np.arange(len(columns)))
    axes[1].set_yticks(np.arange(len(rows)))
    axes[1].set_xticklabels(columns)
    axes[1].set_yticklabels(rows)
    for i in range(len(rows)):
        for j in range(len(columns)):
            axes[1].text(j, i, f'{est_err_table[i, j]:.4f}', ha='center', va='center', color='black')
    # Create colorbars
    cbar1 = fig.colorbar(im1, ax=axes[0])
    cbar2 = fig.colorbar(im2, ax=axes[1])
    # Adjust layout
    plt.tight_layout()
    plt.show()


def Scenario5_table(test_err_table, approx_err_table, est_err_table):
    columns = ['M = 1', 'M = 10', 'M = 50', 'M = 1000']
    rows = ['K = 2', 'K = 5', 'K = 10', 'K = 50']
    # Create test error table
    fig, ax = plt.subplots()
    ax.axis('off')  # Turn off axis labels
    # Create the table
    table_data = [[f'{test_err_table[i, j]:.4f}' for j in range(4)] for i in range(4)]
    table = ax.table(cellText=table_data, colLabels=columns, rowLabels=rows, loc='center',
                     cellLoc='center', cellColours=None)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    # Add title close to the table
    plt.title('ERM approximation errors', y=1.02, pad=-100, fontsize=12, ha='center', va='bottom')
    plt.show()
    # Create approximation/estimation error table
    fig, ax = plt.subplots()
    # Hide the axes
    ax.set_frame_on(False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    # Create a table and add it to the figure
    table_data = []
    for i, row_label in enumerate(rows):
        row_data = []
        for j, col_label in enumerate(columns):
            # Combine the values from both arrays
            cell_text = f'({approx_err_table[i, j]:.4f}, {est_err_table[i, j]:.4f})'
            row_data.append(cell_text)
        table_data.append(row_data)
    table = ax.table(cellText=table_data, rowLabels=rows, colLabels=columns,
                     loc='center')
    # Adjust the cell padding
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    plt.title('Approximation and Estimation errors', y=1.02, pad=-90, fontsize=12, ha='center',
              va='bottom')  # Adjust y value
    plt.show()


def Scenario_6(train_set, test_set):
    """
    Question 6.
    The Bias-Variance Tradeoff.
    You may change the input & output parameters of the function as you wish.
    """
    ############### YOUR CODE GOES HERE ###############
    print("5 prophets [0.3,0.6] error rates, 10 games")
    Scenario_3(train_set, test_set, 5, 10, 0.3, 0.6)
    print("500 prophets [0.25,0.6] error rates, 10 games")
    Scenario_3(train_set, test_set, 500, 10, 0.25, 0.6)


if __name__ == '__main__':
    np.random.seed(0)  # DO NOT MOVE / REMOVE THIS CODE LINE!

    # train, validation and test splits for Scenario 1-3, 5
    # 100 sets, 1000 games in each set
    train_set = create_data(100, 1000)
    # one set, 1000 games in set
    test_set = create_data(1, 1000)[0]

    print(f'Scenario 1 Results:')
    Scenario_1(train_set, test_set, 1)

    print(f'Scenario 2 Results:')
    Scenario_2(train_set, test_set, 10)

    print(f'Scenario 3 Results:')
    Scenario_3(train_set, test_set, 500, 10, 0, 1)

    print(f'Scenario 4 Results:')
    Scenario_4(train_set, test_set, 500, 1000, 0, 1)

    print(f'Scenario 5 Results:')
    Scenario_5(train_set, test_set)

    print(f'Scenario 6 Results:')
    Scenario_6(train_set, test_set)
