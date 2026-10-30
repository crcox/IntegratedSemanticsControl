#!/usr/bin/env python

# %% In[5]:

import os
import warnings

import numpy as np
import pandas as pd
import pingouin as pg
import plotly.express as px
import torch
from scipy.stats import ttest_rel

import analysis
import data
import models
from utils import set_torch_device

warnings.filterwarnings("ignore")

model_path = "models"
num_models = 10
num_bootstrap_sims = 10000
num_training_epochs = 30
num_training_epochs_comparison = 10
train_model = False
train_comparison_model = False

num_objects = 350
num_tasks = 36
num_task_context_units = 16
num_context_independent_units = 64
num_context_dependent_units = 128
size_idx = 2541  # Index of the "is_small" feature
size_task_idx = 33  # Index of the size task

device = set_torch_device()

isc_models = models.load_isc_models(
    num_models=num_models,
    retrain_models=False,
    num_training_epochs=num_training_epochs,
    model_path=model_path,
    device=device,
)

# %% In[7]:


# Extract task context, context independent, and context dependent representations from the models
# Calculate similarity matrices for each representation

task_context_reps = np.zeros((num_models, num_tasks, num_task_context_units))
context_independent_reps = np.zeros(
    (num_models, num_objects, num_context_independent_units)
)
context_dependent_reps = np.zeros(
    (num_models, num_tasks, num_objects, num_context_dependent_units)
)
feature_reps = data.load_feature_reps()

task_context_sim = np.zeros((num_models, num_tasks, num_tasks))
context_independent_sim = np.zeros((num_models, num_objects, num_objects))
context_dependent_sim = np.zeros((num_models, num_tasks, num_objects, num_objects))
feature_sim = analysis.calculate_similarity_matrix(feature_reps, method="correlation")
for model_idx, isc_model in enumerate(isc_models):
    task_context_reps[model_idx] = isc_model.get_task_context_reps()
    context_independent_reps[model_idx] = isc_model.get_context_independent_reps()
    context_dependent_reps[model_idx] = isc_model.get_context_dependent_reps()
    task_context_sim[model_idx] = analysis.calculate_similarity_matrix(
        task_context_reps[model_idx]
    )
    context_independent_sim[model_idx] = analysis.calculate_similarity_matrix(
        context_independent_reps[model_idx]
    )
    context_dependent_sim[model_idx] = analysis.calculate_similarity_matrix(
        context_dependent_reps[model_idx]
    )


# # Part 1: Influence of Control on Semantic Structure

# ## Experiment 1: Empirical Evaluation of Learned Conceptual Structure

# %% In[9]:


stimuli = pd.read_csv("data/leuven_triplets_meanagree.csv")
stimuli.head()


# ## Analysis 2: Operation of Control on Abstract Semantic Dimensions

# %% In[10]:


contexts_to_keep, objects_to_keep, num_objects_to_keep = analysis.filter_nan_similarity(
    feature_reps[:-1], feature_sim[:-1]
)  # exclude naming context

independent_correlations, dependent_correlations = analysis.fast_rsa_bootstrap(
    feature_sim[contexts_to_keep],
    np.repeat(
        context_independent_sim.mean(axis=0)[np.newaxis, :],
        len(contexts_to_keep),
        axis=0,
    ),
    context_dependent_sim.mean(axis=0)[contexts_to_keep],
    objects_to_keep,
    num_objects_to_keep,
    n_sims=num_bootstrap_sims,
)
print(
    f"Context independent vs feature representation r={independent_correlations.mean():.2f}"
)
print(
    f"Context dependent vs feature representation r={dependent_correlations.mean():.2f}"
)
print(
    f"Difference is significant at p={((independent_correlations - dependent_correlations).mean(axis=-1) > 0).mean()}"
)


# ## Experiment 2: Empirical Evaluation of Task-specific Conceptual Structure

# %% In[11]:


stimuli = pd.read_csv("data/rt_rotated_2D_embeddings_and_metadata.csv")
stimuli.head()


# ## Analysis 3: Substructure of Task-Specific Representations

# ### Correlation of Size with Other Features (by Category)

# %% In[13]:


feature_values = data.get_raw_feature_values().sum(axis=0)
animal_idxs = data.get_item_indices_by_category(["bird", "fish", "mammal", "reptile"])
instrument_idxs = data.get_item_indices_by_category(["instrument"])

animal_fts, instrument_fts = (
    feature_values[animal_idxs],
    feature_values[instrument_idxs],
)
animal_sizes, instrument_sizes = (
    feature_values[animal_idxs, size_idx],
    feature_values[instrument_idxs, size_idx],
)
ft_mask = (animal_fts.sum(axis=0) > 0) | (instrument_fts.sum(axis=0) > 0)
animal_fts, instrument_fts = animal_fts[:, ft_mask], instrument_fts[:, ft_mask]

within_corrs, across_corrs = analysis.fast_correlation_splithalf(
    animal_sizes,
    animal_fts,
    instrument_sizes,
    instrument_fts,
    num_bootstrap_sims,
    replace_nan=0,
)
print(f"Correlation within category r={within_corrs.mean():.2f}")
print(f"Correlation between categories r={across_corrs.mean():.2f}")
print(f"Difference is significant at p={((across_corrs - within_corrs) > 0).mean()}")


# ### Angle between Animal and Instrument Size Representations

# %% In[14]:


small_animal_idxs = data.get_item_indices_by_category_and_size(
    ["bird", "fish", "mammal", "reptile"], "small"
)
large_animal_idxs = data.get_item_indices_by_category_and_size(
    ["bird", "fish", "mammal", "reptile"], "large"
)
small_instrument_idxs = data.get_item_indices_by_category_and_size(
    ["instrument"], "small"
)
large_instrument_idxs = data.get_item_indices_by_category_and_size(
    ["instrument"], "large"
)

within_angles, across_angles = (
    np.zeros((num_models, num_bootstrap_sims)),
    np.zeros((num_models, num_bootstrap_sims)),
)
for i in range(num_models):
    size_context_reps = isc_models[i].get_context_dependent_reps()[-3]
    small_animal_reps = size_context_reps[small_animal_idxs]
    large_animal_reps = size_context_reps[large_animal_idxs]
    small_instrument_reps = size_context_reps[small_instrument_idxs]
    large_instrument_reps = size_context_reps[large_instrument_idxs]

    within_angles[i], across_angles[i] = analysis.cosine_splithalf(
        small_animal_reps,
        large_animal_reps,
        small_instrument_reps,
        large_instrument_reps,
        n_sims=num_bootstrap_sims,
    )

print(f"Angle within category={within_angles.mean():.2f}")
print(f"Angle between categories={across_angles.mean():.2f}")
print(f"Difference is significant at p={((within_angles - across_angles) > 0).mean()}")


# #### Comparison Model

# %% In[15]:


comparison_models = models.load_comparison_models(num_models)


# %% In[16]:


small_animal_idxs = data.get_item_indices_by_category_and_size(
    ["bird", "fish", "mammal", "reptile"], "small"
)
large_animal_idxs = data.get_item_indices_by_category_and_size(
    ["bird", "fish", "mammal", "reptile"], "large"
)
small_instrument_idxs = data.get_item_indices_by_category_and_size(
    ["instrument"], "small"
)
large_instrument_idxs = data.get_item_indices_by_category_and_size(
    ["instrument"], "large"
)

small_animal_idxs_complement = [i + num_objects for i in small_animal_idxs]
large_animal_idxs_complement = [i + num_objects for i in large_animal_idxs]
small_instrument_idxs_complement = [i + num_objects for i in small_instrument_idxs]
large_instrument_idxs_complement = [i + num_objects for i in large_instrument_idxs]

within_angles, across_angles = (
    np.zeros((num_models, num_bootstrap_sims)),
    np.zeros((num_models, num_bootstrap_sims)),
)
for i in range(num_models):
    size_context_reps = comparison_models[i].get_context_dependent_reps()[-3]
    small_animal_reps = size_context_reps[small_animal_idxs]
    large_animal_reps = size_context_reps[large_animal_idxs]
    small_instrument_reps = size_context_reps[small_instrument_idxs]
    large_instrument_reps = size_context_reps[large_instrument_idxs]
    small_animal_reps_complement = size_context_reps[small_animal_idxs_complement]
    large_animal_reps_complement = size_context_reps[large_animal_idxs_complement]
    small_instrument_reps_complement = size_context_reps[
        small_instrument_idxs_complement
    ]
    large_instrument_reps_complement = size_context_reps[
        large_instrument_idxs_complement
    ]

    within_angles[i], across_angles[i] = analysis.cosine_splithalf_paired(
        small_animal_reps,
        large_animal_reps,
        small_instrument_reps,
        large_instrument_reps,
        small_animal_reps_complement,
        large_animal_reps_complement,
        small_instrument_reps_complement,
        large_instrument_reps_complement,
        n_sims=num_bootstrap_sims,
    )

print(f"Angle within category={within_angles.mean():.2f}")
print(f"Angle between categories={across_angles.mean():.2f}")
print(f"Difference is significant at p={((within_angles - across_angles) > 0).mean()}")


# # Part 2: Influence of Semantics on Control

# ## Analysis 4: Cross-task Similarity of Representations Used for Control

# %% In[17]:


contexts_to_keep, objects_to_keep, num_objects_to_keep = analysis.filter_nan_similarity(
    feature_reps[:-1], feature_sim[:-1]
)  # exclude naming context
task_ft_sim = analysis.calculate_task_similarity_matrix(
    feature_sim, contexts_to_keep, objects_to_keep, num_objects_to_keep
)

task_context_sim_ = task_context_sim.mean(axis=0)[contexts_to_keep][
    :, contexts_to_keep
][np.triu_indices(len(contexts_to_keep), 1)]

correlation, null_correlations = analysis.fast_correlation_vec_vec_bootstrap(
    task_ft_sim, task_context_sim_, num_bootstrap_sims
)
print(f"Correlation between task-context reps and task similarity r={correlation:.3f}")
print(f"Correlation of the null distribution r={null_correlations.mean():.3f}")
print(
    f"Difference is significant at p={((null_correlations - correlation) > 0).mean()}"
)


# ## Analysis 5: Shaping of Representations Used for Control to Optimize Task Performance

# %% In[18]:


(
    train_x,
    train_y,
    experiment_stimulus_indices,
    small_animal_idxs,
    large_animal_idxs,
    small_instrument_idxs,
    large_instrument_idxs,
) = data.make_full_distractor_training_data()
interleaved_x = (train_x[0].clone(), train_x[1].clone())
interleaved_x[1][:, 0] = 0
interleaved_x[1][:, 1] = 1


# %% In[19]:


angles_blocked = []
for model_idx in range(10):
    learned_context_model = models.ISCModel(
        device="mps",
        num_tasks=2,
        num_task_context_units=16,
        num_context_dependent_hidden_units=128,
    )
    learned_context_model.load_old_model_weights(
        isc_models[0].state_dict(), use_old_size_starting_point=True
    )
    learned_context_model.metrics.append(models.SizeOrthogonalityMetric())
    for epoch in range(5):
        learned_context_model.train(train_x, train_y, epochs=1)
    angles_blocked.append(learned_context_model.metrics[2].values)


# %% In[20]:


tprs = learned_context_model.metrics[1].values
plot_angles = pd.DataFrame(learned_context_model.metrics[2].values)
plot_angles = plot_angles[plot_angles.color == "across"].y.values
ys = list(tprs) + list(plot_angles)
xs = list(range(len(tprs))) * 2
colors = ["True Positive Rate"] * len(tprs) + ["Cosine Distance"] * len(plot_angles)
fig = px.line(
    x=xs,
    y=ys,
    color=colors,
)
fig.update_layout(
    width=1600, height=300, plot_bgcolor="white", legend=dict(font=dict(size=16))
)
fig.update_xaxes(
    title="Epoch",
    showline=True,
    linewidth=1.5,
    linecolor="black",
    mirror=True,
    ticks="outside",
    showgrid=True,
    gridcolor="#dddddd",
    titlefont=dict(size=20),
)
fig.update_yaxes(
    title="Value",
    showline=True,
    linewidth=1.5,
    linecolor="black",
    mirror=True,
    ticks="outside",
    showgrid=True,
    gridcolor="#dddddd",
    titlefont=dict(size=20),
)
fig.show()


# %% In[21]:


angles_interleaved = []
for model_idx in range(10):
    learned_context_model = models.ISCModel(
        device="mps",
        num_tasks=2,
        num_task_context_units=16,
        num_context_dependent_hidden_units=128,
    )
    learned_context_model.load_old_model_weights(
        isc_models[0].state_dict(), use_old_size_starting_point=True
    )
    learned_context_model.metrics.append(models.SizeOrthogonalityMetric())
    for epoch in range(5):
        learned_context_model.train(interleaved_x, train_y, epochs=1)
    angles_interleaved.append(learned_context_model.metrics[2].values)


# %% In[22]:


for name, angles in zip(
    ["blocked", "interleaved"], [angles_blocked, angles_interleaved]
):
    df = None
    for i in range(10):
        d = pd.DataFrame(angles[i])
        d["model"] = i
        if df is None:
            df = d
        else:
            df = pd.concat([df, d])
    win_start = df[(df.x == 0) & (df.color == "within")].y.values
    win_end = df[(df.x == 9) & (df.color == "within")].y.values
    acr_start = df[(df.x == 0) & (df.color == "across")].y.values
    acr_end = df[(df.x == 9) & (df.color == "across")].y.values
    print(f"-----{name}-----")
    print(f"Within from {win_start.mean():.2f} to {win_end.mean():.2f}")
    print(f"Across from {acr_start.mean():.2f} to {acr_end.mean():.2f}")
    print("Ttest within:", ttest_rel(win_start, win_end, alternative="greater"))
    print("Ttest across:", ttest_rel(acr_start, acr_end, alternative="less"))
    print()


# ## Experiment 3: Effects of Semantics on Representations Used for Control

# ### Model Training

# %% In[24]:


train_x, train_y, size_conditions, cat_conditions, random_cat_conditions, blocks = (
    data.make_behavioral_experiment_training_data(distractor_strength=0.975)
)


def calc_model_error(model, train_x, train_y, noise=0):
    errors = torch.abs(model(train_x, noise=noise) - train_y)[:, [2541, 2542]].mean(
        axis=-1
    )
    return errors.cpu().detach().numpy()


error_data = []
for model_idx in range(1):
    simulation_model = models.ISCModel(
        device="mps",
        num_tasks=5,
        num_task_context_units=16,
        num_context_dependent_hidden_units=128,
    )
    save_file = f"isc_model-{model_idx}-bhvsim.torch"
    if save_file in os.listdir("models"):
        simulation_model.load_state_dict(torch.load(os.path.join("models", save_file)))
    else:
        simulation_model.load_old_model_weights(
            isc_models[model_idx].state_dict(), use_old_size_starting_point=True
        )
        errors = calc_model_error(simulation_model, train_x, train_y)
        c = 1
        while errors.mean() > 0.18:
            simulation_model.train(train_x, train_y, epochs=1)
            errors = calc_model_error(simulation_model, train_x, train_y)
            c += 1
        print(model_idx, errors.mean(), c)
        torch.save(simulation_model.state_dict(), os.path.join("models", save_file))
    errors = calc_model_error(simulation_model, train_x, train_y)
    accs = (
        (
            (simulation_model(train_x)[:, 2541] > simulation_model(train_x)[:, 2542])
            == train_y[:, 2541]
        )
        .float()
        .cpu()
        .detach()
        .numpy()
    )
    error_data.append(
        pd.DataFrame(
            {
                "model": [model_idx] * len(errors),
                "rt": errors,
                "error": 1 - accs,
                "size_condition": size_conditions,
                "cat_condition": cat_conditions,
                "rand_condition": random_cat_conditions,
                "block_type": blocks,
            }
        )
    )
error_data = pd.concat(error_data, axis=0)


# %% In[27]:


error_data = []
for i in range(71):
    errors = calc_model_error(simulation_model, train_x, train_y, noise=1.2)
    preds = simulation_model(train_x, noise=1.175)
    accs = (
        ((preds[:, 2541] > preds[:, 2542]) == train_y[:, 2541])
        .float()
        .cpu()
        .detach()
        .numpy()
    )
    error_data.append(
        pd.DataFrame(
            {
                "model": [i] * len(errors),
                "rt": errors,
                "error": 1 - accs,
                "size_condition": size_conditions,
                "cat_condition": cat_conditions,
                "rand_condition": random_cat_conditions,
                "block_type": blocks,
            }
        )
    )
error_data = pd.concat(error_data, axis=0)
# error_data.to_csv('data/experiment3_simulation_data_0200.csv')


# ### Model Performance Statistics

# %% In[28]:


model_data = pd.read_csv(
    "data/experiment3_simulation_data_0200.csv"
)  # models that were trained to 0.200 avg error

model_data = model_data[model_data.block_type != "random"]
model_data_interleaved = model_data[model_data.block_type == "interleaved"]
model_data_blocked = model_data[model_data.block_type == "blocked"]

print("Blocked ANOVA")
print("  RT")
display(
    pg.rm_anova(
        data=model_data_blocked,
        dv="rt",
        within=["size_condition", "cat_condition"],
        subject="model",
        detailed=True,
        effsize="n2",
    )
)
print("  Error")
display(
    pg.rm_anova(
        data=model_data_blocked,
        dv="error",
        within=["size_condition", "cat_condition"],
        subject="model",
        detailed=True,
        effsize="n2",
    )
)
print()
print("Interleaved ANOVA")
print("  RT")
display(
    pg.rm_anova(
        data=model_data_interleaved,
        dv="rt",
        within=["size_condition", "cat_condition"],
        subject="model",
        detailed=True,
        effsize="n2",
    )
)
print("  Error")
display(
    pg.rm_anova(
        data=model_data_interleaved,
        dv="error",
        within=["size_condition", "cat_condition"],
        subject="model",
        detailed=True,
        effsize="n2",
    )
)


# %% In[29]:


print("Blocked Simple Effects - RT")
print("    Category Effect @ Size Match")
display(
    pg.rm_anova(
        data=model_data_blocked[model_data_blocked.size_condition == "s_ma"],
        dv="rt",
        within="cat_condition",
        subject="model",
        detailed=True,
        effsize="n2",
    )
)
print("    Category Effect @ Size Mismatch")
display(
    pg.rm_anova(
        data=model_data_blocked[model_data_blocked.size_condition == "s_ms"],
        dv="rt",
        within="cat_condition",
        subject="model",
        detailed=True,
        effsize="n2",
    )
)
print("Blocked Simple Effects - Error")
print("    Category Effect @ Size Match")
display(
    pg.rm_anova(
        data=model_data_blocked[model_data_blocked.size_condition == "s_ma"],
        dv="error",
        within="cat_condition",
        subject="model",
        detailed=True,
        effsize="n2",
    )
)
print("    Category Effect @ Size Mismatch")
display(
    pg.rm_anova(
        data=model_data_blocked[model_data_blocked.size_condition == "s_ms"],
        dv="error",
        within="cat_condition",
        subject="model",
        detailed=True,
        effsize="n2",
    )
)


# %% In[30]:


print("Blocked vs Interleaved Interference Effect Size")
transformed_data = (
    model_data[model_data.size_condition == "s_ms"]
    .groupby(["block_type", "model", "cat_condition"], as_index=False)
    .mean()
    .pivot_table(index="model", columns=["block_type", "cat_condition"], values="rt")
)
blocked_effect_size = (
    transformed_data["blocked", "c_ma"] - transformed_data["blocked", "c_ms"]
)
interleaved_effect_size = (
    transformed_data["interleaved", "c_ma"] - transformed_data["interleaved", "c_ms"]
)
ttest_result = ttest_rel(blocked_effect_size.values, interleaved_effect_size.values)
print(
    f"RT Means: Blocked={blocked_effect_size.mean():.2f}, Interleaved={interleaved_effect_size.mean():.2f}, t={ttest_result.statistic:.2f}, p={ttest_result.pvalue:.3f}"
)
transformed_data = (
    model_data[model_data.size_condition == "s_ms"]
    .groupby(["block_type", "model", "cat_condition"], as_index=False)
    .mean()
    .pivot_table(index="model", columns=["block_type", "cat_condition"], values="error")
)
blocked_effect_size = (
    transformed_data["blocked", "c_ma"] - transformed_data["blocked", "c_ms"]
)
interleaved_effect_size = (
    transformed_data["interleaved", "c_ma"] - transformed_data["interleaved", "c_ms"]
)
ttest_result = ttest_rel(blocked_effect_size.values, interleaved_effect_size.values)
print(
    f"Error Means: Blocked={blocked_effect_size.mean():.2f}, Interleaved={interleaved_effect_size.mean():.2f}, t={ttest_result.statistic:.2f}, p={ttest_result.pvalue:.3f}"
)


# ### Behavioral Performance Statistics

# %% In[32]:


behavioral_data = data.load_behavioral_data()
behavioral_data["rt"] = np.log(behavioral_data.rt.values)

acc_data = behavioral_data[behavioral_data.participant_type != "random"]
acc_data = acc_data.groupby(
    ["block_type_agg", "size_condition", "cat_condition", "participant"], as_index=False
).mean()
acc_data_interleaved = acc_data[acc_data.block_type_agg == "interleaved"]
acc_data_blocked = acc_data[acc_data.block_type_agg == "blocked"]

rt_data = behavioral_data[
    (behavioral_data.participant_type != "random") & (behavioral_data.correct == 1)
]
rt_data = rt_data.groupby(
    ["block_type_agg", "size_condition", "cat_condition", "participant"], as_index=False
).mean()
rt_data_interleaved = rt_data[rt_data.block_type_agg == "interleaved"]
rt_data_blocked = rt_data[rt_data.block_type_agg == "blocked"]

print("Interleaved ANOVA")
print("  RT")
display(
    pg.rm_anova(
        data=rt_data_interleaved,
        dv="rt",
        within=["size_condition", "cat_condition"],
        subject="participant",
        detailed=True,
        effsize="n2",
    )
)
print("  Acc")
display(
    pg.rm_anova(
        data=acc_data_interleaved,
        dv="correct",
        within=["size_condition", "cat_condition"],
        subject="participant",
        detailed=True,
        effsize="n2",
    )
)
print()
print("Blocked ANOVA")
print("  RT")
display(
    pg.rm_anova(
        data=rt_data_blocked,
        dv="rt",
        within=["size_condition", "cat_condition"],
        subject="participant",
        detailed=True,
        effsize="n2",
    )
)
print("  Acc")
display(
    pg.rm_anova(
        data=acc_data_blocked,
        dv="correct",
        within=["size_condition", "cat_condition"],
        subject="participant",
        detailed=True,
        effsize="n2",
    )
)
print()


# %% In[33]:


print("Blocked Simple Effects")
print("  RT")
print("    Category Effect @ Size Match")
display(
    pg.rm_anova(
        data=rt_data_blocked[rt_data_blocked.size_condition == "s_ma"],
        dv="rt",
        within="cat_condition",
        subject="participant",
        detailed=True,
        effsize="n2",
    )
)
print("    Category Effect @ Size Mismatch")
display(
    pg.rm_anova(
        data=rt_data_blocked[rt_data_blocked.size_condition == "s_ms"],
        dv="rt",
        within="cat_condition",
        subject="participant",
        detailed=True,
        effsize="n2",
    )
)
print("  Acc")
print("    Category Effect @ Size Match")
display(
    pg.rm_anova(
        data=acc_data_blocked[acc_data_blocked.size_condition == "s_ma"],
        dv="correct",
        within="cat_condition",
        subject="participant",
        detailed=True,
        effsize="n2",
    )
)
print("    Category Effect @ Size Mismatch")
display(
    pg.rm_anova(
        data=acc_data_blocked[acc_data_blocked.size_condition == "s_ms"],
        dv="correct",
        within="cat_condition",
        subject="participant",
        detailed=True,
        effsize="n2",
    )
)


# %% In[34]:


from scipy.stats import ttest_rel

print("Blocked vs Interleaved Interference Effect Size")
transformed_data = (
    rt_data[rt_data.size_condition == "s_ms"]
    .groupby(["block_type_agg", "participant", "cat_condition"], as_index=False)
    .mean()
    .pivot_table(
        index="participant", columns=["block_type_agg", "cat_condition"], values="rt"
    )
)
blocked_effect_size = (
    transformed_data["blocked", "c_ma"] - transformed_data["blocked", "c_ms"]
)
interleaved_effect_size = (
    transformed_data["interleaved", "c_ma"] - transformed_data["interleaved", "c_ms"]
)
ttest_result = ttest_rel(blocked_effect_size.values, interleaved_effect_size.values)
print(
    f"RT - means: Blocked={blocked_effect_size.mean():.2f}, Interleaved={interleaved_effect_size.mean():.2f}, t={ttest_result.statistic:.2f}, p={ttest_result.pvalue:.3f}"
)

transformed_data = (
    acc_data[acc_data.size_condition == "s_ms"]
    .groupby(["block_type_agg", "participant", "cat_condition"], as_index=False)
    .mean()
    .pivot_table(
        index="participant",
        columns=["block_type_agg", "cat_condition"],
        values="correct",
    )
)
blocked_effect_size = (
    transformed_data["blocked", "c_ma"] - transformed_data["blocked", "c_ms"]
)
interleaved_effect_size = (
    transformed_data["interleaved", "c_ma"] - transformed_data["interleaved", "c_ms"]
)
ttest_result = ttest_rel(blocked_effect_size.values, interleaved_effect_size.values)
print(
    f"Acc - means: Blocked={blocked_effect_size.mean():.2f}, Interleaved={interleaved_effect_size.mean():.2f}, t={ttest_result.statistic:.2f}, p={ttest_result.pvalue:.3f}"
)


# ### Plots

# %% In[35]:


behavioral_data = data.load_behavioral_data()
behavioral_data["rt"] = np.log(behavioral_data.rt.values)

acc_data = behavioral_data[behavioral_data.participant_type != "random"]
acc_data = acc_data.groupby(
    ["block_type_agg", "size_condition", "cat_condition", "participant"], as_index=False
).mean()
acc_data_interleaved = acc_data[acc_data.block_type_agg == "interleaved"]
acc_data_blocked = acc_data[acc_data.block_type_agg == "blocked"]

rt_data = behavioral_data[
    (behavioral_data.participant_type != "random") & (behavioral_data.correct == 1)
]
rt_data = rt_data.groupby(
    ["block_type_agg", "size_condition", "cat_condition", "participant"], as_index=False
).mean()
rt_data_interleaved = rt_data[rt_data.block_type_agg == "interleaved"]
rt_data_blocked = rt_data[rt_data.block_type_agg == "blocked"]

plot_data_interleaved = data.add_within_subject_error_bars(rt_data_interleaved)
plot_data_interleaved = plot_data_interleaved.groupby(
    ["cat_condition", "size_condition"], as_index=False
).mean()
plot_data_interleaved["cat_condition"] = plot_data_interleaved.cat_condition.replace(
    {"c_ma": "Category Match", "c_ms": "Category Mismatch"}
)
plot_data_interleaved["size_condition"] = plot_data_interleaved.size_condition.replace(
    {"s_ma": "Size Match", "s_ms": "Size Mismatch"}
)
plot_data_blocked = data.add_within_subject_error_bars(rt_data_blocked)
plot_data_blocked = plot_data_blocked.groupby(
    ["cat_condition", "size_condition"], as_index=False
).mean()
plot_data_blocked["cat_condition"] = plot_data_blocked.cat_condition.replace(
    {"c_ma": "Category Match", "c_ms": "Category Mismatch"}
)
plot_data_blocked["size_condition"] = plot_data_blocked.size_condition.replace(
    {"s_ma": "Size Match", "s_ms": "Size Mismatch"}
)

model_data_interleaved = model_data_interleaved.groupby(
    ["model", "cat_condition", "size_condition"], as_index=False
).mean()
model_data_blocked = model_data_blocked.groupby(
    ["model", "cat_condition", "size_condition"], as_index=False
).mean()
plot_data_interleaved_model = data.add_within_subject_error_bars(
    model_data_interleaved, subject="model", dv="rt", remove_mean=True
)
plot_data_interleaved_model = plot_data_interleaved_model.groupby(
    ["cat_condition", "size_condition"], as_index=False
).mean()
plot_data_interleaved_model["cat_condition"] = (
    plot_data_interleaved_model.cat_condition.replace(
        {"c_ma": "Category Match", "c_ms": "Category Mismatch"}
    )
)
plot_data_interleaved_model["size_condition"] = (
    plot_data_interleaved_model.size_condition.replace(
        {"s_ma": "Size Match", "s_ms": "Size Mismatch"}
    )
)
plot_data_blocked_model = data.add_within_subject_error_bars(
    model_data_blocked, subject="model", dv="rt", remove_mean=True
)
plot_data_blocked_model = plot_data_blocked_model.groupby(
    ["cat_condition", "size_condition"], as_index=False
).mean()
plot_data_blocked_model["cat_condition"] = (
    plot_data_blocked_model.cat_condition.replace(
        {"c_ma": "Category Match", "c_ms": "Category Mismatch"}
    )
)
plot_data_blocked_model["size_condition"] = (
    plot_data_blocked_model.size_condition.replace(
        {"s_ma": "Size Match", "s_ms": "Size Mismatch"}
    )
)
plot_data_interleaved_acc_model = data.add_within_subject_error_bars(
    model_data_interleaved, subject="model", dv="error", remove_mean=True
)
plot_data_interleaved_acc_model = plot_data_interleaved_acc_model.groupby(
    ["cat_condition", "size_condition"], as_index=False
).mean()
plot_data_interleaved_acc_model["cat_condition"] = (
    plot_data_interleaved_acc_model.cat_condition.replace(
        {"c_ma": "Category Match", "c_ms": "Category Mismatch"}
    )
)
plot_data_interleaved_acc_model["size_condition"] = (
    plot_data_interleaved_acc_model.size_condition.replace(
        {"s_ma": "Size Match", "s_ms": "Size Mismatch"}
    )
)
plot_data_blocked_acc_model = data.add_within_subject_error_bars(
    model_data_blocked, subject="model", dv="error", remove_mean=True
)
plot_data_blocked_acc_model = plot_data_blocked_acc_model.groupby(
    ["cat_condition", "size_condition"], as_index=False
).mean()
plot_data_blocked_acc_model["cat_condition"] = (
    plot_data_blocked_acc_model.cat_condition.replace(
        {"c_ma": "Category Match", "c_ms": "Category Mismatch"}
    )
)
plot_data_blocked_acc_model["size_condition"] = (
    plot_data_blocked_acc_model.size_condition.replace(
        {"s_ma": "Size Match", "s_ms": "Size Mismatch"}
    )
)


acc_data_interleaved["error"] = 1 - acc_data_interleaved.correct
plot_data_interleaved_acc = data.add_within_subject_error_bars(
    acc_data_interleaved, dv="error", remove_mean=True
)
plot_data_interleaved_acc = plot_data_interleaved_acc.groupby(
    ["cat_condition", "size_condition"], as_index=False
).mean()
plot_data_interleaved_acc["cat_condition"] = (
    plot_data_interleaved_acc.cat_condition.replace(
        {"c_ma": "Category Match", "c_ms": "Category Mismatch"}
    )
)
plot_data_interleaved_acc["size_condition"] = (
    plot_data_interleaved_acc.size_condition.replace(
        {"s_ma": "Size Match", "s_ms": "Size Mismatch"}
    )
)

acc_data_blocked["error"] = 1 - acc_data_blocked.correct
plot_data_blocked_acc = data.add_within_subject_error_bars(
    acc_data_blocked, dv="error", remove_mean=True
)
plot_data_blocked_acc = plot_data_blocked_acc.groupby(
    ["cat_condition", "size_condition"], as_index=False
).mean()
plot_data_blocked_acc["cat_condition"] = plot_data_blocked_acc.cat_condition.replace(
    {"c_ma": "Category Match", "c_ms": "Category Mismatch"}
)
plot_data_blocked_acc["size_condition"] = plot_data_blocked_acc.size_condition.replace(
    {"s_ma": "Size Match", "s_ms": "Size Mismatch"}
)

import plotly.express as px
from plotly.subplots import make_subplots

model_rt_interleaved = px.bar(
    plot_data_interleaved_model,
    x="size_condition",
    color="cat_condition",
    y="rt_normalized",
    barmode="group",
    error_y="rt_error",
)
human_rt_interleaved = px.bar(
    plot_data_interleaved,
    x="size_condition",
    color="cat_condition",
    y="rt_normalized",
    barmode="group",
    error_y="rt_error",
)
model_rt_blocked = px.bar(
    plot_data_blocked_model,
    x="size_condition",
    color="cat_condition",
    y="rt_normalized",
    barmode="group",
    error_y="rt_error",
)
human_rt_blocked = px.bar(
    plot_data_blocked,
    x="size_condition",
    color="cat_condition",
    y="rt_normalized",
    barmode="group",
    error_y="rt_error",
)
human_acc_interleaved = px.bar(
    plot_data_interleaved_acc,
    x="size_condition",
    color="cat_condition",
    y="error_normalized",
    barmode="group",
    error_y="error_error",
)
human_acc_blocked = px.bar(
    plot_data_blocked_acc,
    x="size_condition",
    color="cat_condition",
    y="error_normalized",
    barmode="group",
    error_y="error_error",
)
model_acc_interleaved = px.bar(
    plot_data_interleaved_acc_model,
    x="size_condition",
    color="cat_condition",
    y="error_normalized",
    barmode="group",
    error_y="error_error",
)
model_acc_blocked = px.bar(
    plot_data_blocked_acc_model,
    x="size_condition",
    color="cat_condition",
    y="error_normalized",
    barmode="group",
    error_y="error_error",
)

fig = make_subplots(
    rows=2,
    cols=2,
    row_titles=["Categorically Blocked Condition", "Interleaved Condition"],
    column_titles=["Model Performance", "Human Performance"],
)
fig.add_trace(model_rt_interleaved.data[0], row=2, col=1)
fig.add_trace(model_rt_interleaved.data[1], row=2, col=1)
fig.add_trace(human_rt_interleaved.data[0], row=2, col=2)
fig.add_trace(human_rt_interleaved.data[1], row=2, col=2)
fig.add_trace(model_rt_blocked.data[0], row=1, col=1)
fig.add_trace(model_rt_blocked.data[1], row=1, col=1)
fig.add_trace(human_rt_blocked.data[0], row=1, col=2)
fig.add_trace(human_rt_blocked.data[1], row=1, col=2)
for idx, trace in enumerate(fig.data):
    if idx > 1:
        trace.showlegend = False
fig.update_layout(
    title="Log Reaction Time (log ms, Relative to Mean) by Condition",
    yaxis_side="left",
    width=1200,
    height=800,
    titlefont=dict(size=20),
)
fig.update_layout(
    plot_bgcolor="white",
    title_x=0.5,
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, font=dict(size=16)),
)
fig.update_xaxes(
    showline=True,
    linewidth=1.5,
    linecolor="black",
    tickfont=dict(size=16),
    mirror=True,
    ticks="outside",
    showgrid=False,
    titlefont=dict(size=20),
)
fig.update_yaxes(
    showline=True,
    linewidth=1.5,
    linecolor="black",
    mirror=True,
    ticks="outside",
    showgrid=False,
    titlefont=dict(size=20),
    zeroline=True,
    zerolinecolor="black",
    zerolinewidth=1,
)
# fig.add_annotation(dict(x=0,y=.25,xref='x1',yref='y1',text='test',showarrow=False))
fig["layout"]["annotations"][2].update(
    text="Categorically Blocked Condition", x=-0.07, textangle=-90
)
fig["layout"]["annotations"][3].update(
    text="Interleaved Condition", x=-0.07, textangle=-90
)
fig.show()

fig = make_subplots(
    rows=2,
    cols=2,
    row_titles=["Categorically Blocked Condition", "Interleaved Condition"],
    column_titles=["Model Performance", "Human Performance"],
)
fig.add_trace(human_acc_interleaved.data[0], row=2, col=2)
fig.add_trace(human_acc_interleaved.data[1], row=2, col=2)
fig.add_trace(model_acc_interleaved.data[0], row=2, col=1)
fig.add_trace(model_acc_interleaved.data[1], row=2, col=1)
fig.add_trace(human_acc_blocked.data[0], row=1, col=2)
fig.add_trace(human_acc_blocked.data[1], row=1, col=2)
fig.add_trace(model_acc_blocked.data[0], row=1, col=1)
fig.add_trace(model_acc_blocked.data[1], row=1, col=1)
for idx, trace in enumerate(fig.data):
    if idx > 1:
        trace.showlegend = False
fig.update_layout(
    title="Error Rate (%, Relative to Mean) by Condition",
    yaxis_side="left",
    width=1200,
    height=800,
    titlefont=dict(size=20),
)
fig.update_layout(
    plot_bgcolor="white",
    title_x=0.5,
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, font=dict(size=16)),
)
fig.update_xaxes(
    showline=True,
    linewidth=1.5,
    linecolor="black",
    tickfont=dict(size=16),
    mirror=True,
    ticks="outside",
    showgrid=False,
    titlefont=dict(size=20),
)
fig.update_yaxes(
    showline=True,
    linewidth=1.5,
    linecolor="black",
    mirror=True,
    ticks="outside",
    showgrid=False,
    titlefont=dict(size=20),
    zeroline=True,
    zerolinecolor="black",
    zerolinewidth=1,
)
# fig.add_annotation(dict(x=0,y=.25,xref='x1',yref='y1',text='test',showarrow=False))
fig["layout"]["annotations"][2].update(
    text="Categorically Blocked Condition", x=-0.07, textangle=-90
)
fig["layout"]["annotations"][3].update(
    text="Interleaved Condition", x=-0.07, textangle=-90
)
fig.show()


# # Supplementary Information

# ## Experiment 3 Control: Response-Set Blocking

# ### Behavioral Performance Statistics

# %% In[39]:


behavioral_data = data.load_behavioral_data()
x = behavioral_data[behavioral_data.participant_type == "random"]
set(x[x.block_type == "random1"].target.values)


# %% In[40]:


behavioral_data = data.load_behavioral_data()
behavioral_data["rt"] = np.log(behavioral_data.rt)

acc_data = behavioral_data[behavioral_data.participant_type == "random"]
acc_data = acc_data.groupby(
    ["block_type_agg", "size_condition", "rand_condition", "participant"],
    as_index=False,
).mean()
acc_data_interleaved = acc_data[acc_data.block_type_agg == "interleaved"]
acc_data_blocked = acc_data[acc_data.block_type_agg == "random"]

rt_data = behavioral_data[
    (behavioral_data.participant_type == "random") & (behavioral_data.correct == 1)
]
rt_data = rt_data.groupby(
    ["block_type_agg", "size_condition", "rand_condition", "participant"],
    as_index=False,
).mean()
rt_data_interleaved = rt_data[rt_data.block_type_agg == "interleaved"]
rt_data_blocked = rt_data[rt_data.block_type_agg == "random"]


print("Response-Set Blocked ANOVA")
print("  RT")
display(
    pg.rm_anova(
        data=rt_data_blocked,
        dv="rt",
        within=["size_condition", "rand_condition"],
        subject="participant",
        detailed=True,
        effsize="n2",
    )
)
print("  Acc")
display(
    pg.rm_anova(
        data=acc_data_blocked,
        dv="correct",
        within=["size_condition", "rand_condition"],
        subject="participant",
        detailed=True,
        effsize="n2",
    )
)
print()


# ### Plots

# %% In[41]:


behavioral_data["cat_condition"] = behavioral_data["rand_condition"]

acc_data = behavioral_data[behavioral_data.participant_type == "random"]
acc_data = acc_data.groupby(
    ["block_type_agg", "size_condition", "cat_condition", "participant"], as_index=False
).mean()
acc_data_interleaved = acc_data[acc_data.block_type_agg == "interleaved"]
acc_data_blocked = acc_data[acc_data.block_type_agg == "random"]

rt_data = behavioral_data[
    (behavioral_data.participant_type == "random") & (behavioral_data.correct == 1)
]
rt_data = rt_data.groupby(
    ["block_type_agg", "size_condition", "cat_condition", "participant"], as_index=False
).mean()
rt_data_interleaved = rt_data[rt_data.block_type_agg == "interleaved"]
rt_data_blocked = rt_data[rt_data.block_type_agg == "random"]

plot_data_interleaved = data.add_within_subject_error_bars(rt_data_interleaved)
plot_data_interleaved = plot_data_interleaved.groupby(
    ["cat_condition", "size_condition"], as_index=False
).mean()
plot_data_interleaved["cat_condition"] = plot_data_interleaved.cat_condition.replace(
    {"a_ma": "Response-Set Match", "a_ms": "Response-Set Mismatch"}
)
plot_data_interleaved["size_condition"] = plot_data_interleaved.size_condition.replace(
    {"s_ma": "Size Match", "s_ms": "Size Mismatch"}
)
plot_data_blocked = data.add_within_subject_error_bars(rt_data_blocked)
plot_data_blocked = plot_data_blocked.groupby(
    ["cat_condition", "size_condition"], as_index=False
).mean()
plot_data_blocked["cat_condition"] = plot_data_blocked.cat_condition.replace(
    {"a_ma": "Response-Set Match", "a_ms": "Response-Set Mismatch"}
)
plot_data_blocked["size_condition"] = plot_data_blocked.size_condition.replace(
    {"s_ma": "Size Match", "s_ms": "Size Mismatch"}
)


acc_data_interleaved["error"] = 1 - acc_data_interleaved.correct
plot_data_interleaved_acc = data.add_within_subject_error_bars(
    acc_data_interleaved, dv="error", remove_mean=True
)
plot_data_interleaved_acc = plot_data_interleaved_acc.groupby(
    ["cat_condition", "size_condition"], as_index=False
).mean()
plot_data_interleaved_acc["cat_condition"] = (
    plot_data_interleaved_acc.cat_condition.replace(
        {"a_ma": "Response-Set Match", "a_ms": "Response-Set Mismatch"}
    )
)
plot_data_interleaved_acc["size_condition"] = (
    plot_data_interleaved_acc.size_condition.replace(
        {"s_ma": "Size Match", "s_ms": "Size Mismatch"}
    )
)

acc_data_blocked["error"] = 1 - acc_data_blocked.correct
plot_data_blocked_acc = data.add_within_subject_error_bars(
    acc_data_blocked, dv="error", remove_mean=True
)
plot_data_blocked_acc = plot_data_blocked_acc.groupby(
    ["cat_condition", "size_condition"], as_index=False
).mean()
plot_data_blocked_acc["cat_condition"] = plot_data_blocked_acc.cat_condition.replace(
    {"a_ma": "Response-Set Match", "a_ms": "Response-Set Mismatch"}
)
plot_data_blocked_acc["size_condition"] = plot_data_blocked_acc.size_condition.replace(
    {"s_ma": "Size Match", "s_ms": "Size Mismatch"}
)

import plotly.express as px
from plotly.subplots import make_subplots

human_rt_interleaved = px.bar(
    plot_data_interleaved,
    x="size_condition",
    color="cat_condition",
    y="rt_normalized",
    barmode="group",
    error_y="rt_error",
)
human_rt_blocked = px.bar(
    plot_data_blocked,
    x="size_condition",
    color="cat_condition",
    y="rt_normalized",
    barmode="group",
    error_y="rt_error",
)
human_acc_interleaved = px.bar(
    plot_data_interleaved_acc,
    x="size_condition",
    color="cat_condition",
    y="error_normalized",
    barmode="group",
    error_y="error_error",
)
human_acc_blocked = px.bar(
    plot_data_blocked_acc,
    x="size_condition",
    color="cat_condition",
    y="error_normalized",
    barmode="group",
    error_y="error_error",
)

fig = make_subplots(
    rows=1,
    cols=2,
    column_titles=["Response-Set Blocked Condition", "Interleaved Condition"],
)
fig.add_trace(human_rt_interleaved.data[0], row=1, col=1)
fig.add_trace(human_rt_interleaved.data[1], row=1, col=1)
fig.add_trace(human_rt_blocked.data[0], row=1, col=2)
fig.add_trace(human_rt_blocked.data[1], row=1, col=2)
for idx, trace in enumerate(fig.data):
    if idx > 1:
        trace.showlegend = False
fig.update_layout(
    title="Log Reaction Time (log ms, Relative to Mean) by Condition",
    yaxis_side="left",
    width=1200,
    height=400,
    titlefont=dict(size=20),
)
fig.update_layout(
    plot_bgcolor="white",
    title_x=0.5,
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, font=dict(size=16)),
)
fig.update_xaxes(
    showline=True,
    linewidth=1.5,
    linecolor="black",
    tickfont=dict(size=16),
    mirror=True,
    ticks="outside",
    showgrid=False,
    titlefont=dict(size=20),
)
fig.update_yaxes(
    showline=True,
    linewidth=1.5,
    linecolor="black",
    mirror=True,
    ticks="outside",
    showgrid=False,
    titlefont=dict(size=20),
    zeroline=True,
    zerolinecolor="black",
    zerolinewidth=1,
)
fig.show()

fig = make_subplots(
    rows=1,
    cols=2,
    column_titles=["Response-Set Blocked Condition", "Interleaved Condition"],
)
fig.add_trace(human_acc_interleaved.data[0], row=1, col=1)
fig.add_trace(human_acc_interleaved.data[1], row=1, col=1)
fig.add_trace(human_acc_blocked.data[0], row=1, col=2)
fig.add_trace(human_acc_blocked.data[1], row=1, col=2)
for idx, trace in enumerate(fig.data):
    if idx > 1:
        trace.showlegend = False
fig.update_layout(
    title="Error Rate (%, Relative to Mean) by Condition",
    yaxis_side="left",
    width=1200,
    height=400,
    titlefont=dict(size=20),
)
fig.update_layout(
    plot_bgcolor="white",
    title_x=0.5,
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, font=dict(size=16)),
)
fig.update_xaxes(
    showline=True,
    linewidth=1.5,
    linecolor="black",
    tickfont=dict(size=16),
    mirror=True,
    ticks="outside",
    showgrid=False,
    titlefont=dict(size=20),
)
fig.update_yaxes(
    showline=True,
    linewidth=1.5,
    linecolor="black",
    mirror=True,
    ticks="outside",
    showgrid=False,
    titlefont=dict(size=20),
    zeroline=True,
    zerolinecolor="black",
    zerolinewidth=1,
)
fig.show()
