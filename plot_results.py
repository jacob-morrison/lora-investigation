from read_results import get_data
import matplotlib.pyplot as plt
import seaborn as sns

TASK = 'case-hold'
# MODEL = 'microsoft/deberta-v3-base'
MODEL = 'microsoft/deberta-v3-large'

data, _ = get_data(TASK)

# TODO: plots to make for each model
# 1. Accuracy as LoRA rank changes
# 2. Maximum accuracy for each finetuning method (full ft + lora x rank)
# 3. Accuracy as LR changes?

fig, axs = plt.subplots(ncols=2)

df_1 = data[data['model'] == MODEL]
df_1 = df_1[df_1['rank'] > -1]

p1 = sns.lineplot(
    data=df_1,
    x='rank',
    y='accuracy',
    hue='model',
    # errorbar=None,
    ax=axs[0],
    # linewidth=line_width
    errorbar=('pi', 100)
)
p1.set(xscale='log')
p1.set(title='LoRA Performance')
p1.set_ylim(bottom=0, top=1.0)

df_2 = data[data['model'] == MODEL]
# df_2 = df_2[df_2['rank'] > 500]

p2 = sns.lineplot(
    data=df_2,
    x='learning rate',
    y='accuracy',
    hue='method and rank',
    # errorbar=None,
    ax=axs[1],
    # linewidth=line_width
    errorbar=('pi', 100)
)
p2.set(xscale='log')
p2.set(title='Performance vs Learning Rate')
p2.set_ylim(bottom=0, top=1.0)
sns.move_legend(axs[1], "lower left")

# df_3 = data[data['model'] == MODEL]
# p3 = sns.lineplot(
#     data=df_3,
#     x='learning rate',
#     y='accuracy',
#     hue='method and rank',
#     # errorbar=None,
#     ax=axs[2],
#     # linewidth=line_width
# )
# p3.set_ylim(bottom=0, top=1.0)

plt.show()
