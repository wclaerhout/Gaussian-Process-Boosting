### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ a93795c0-38e3-11f1-bfa9-8182dfeb44b1
using Pkg; Pkg.activate(".")

# ╔═╡ 05d81ba8-8ba3-44e8-8624-2773438b5944
Pkg.add("XLSX")

# ╔═╡ 6468d1fa-9186-4267-95ab-dd0b493429ac
using XLSX

# ╔═╡ a6aa8b59-54d8-4879-bce7-1e20f5407b1b
using StatsPlots, PlutoUI; TableOfContents()

# ╔═╡ 8bd1c120-9758-4ae6-bea9-e8bb80e2f64b
using MLJ; using EvoTrees

# ╔═╡ ebaf341f-398d-4cbf-8bd5-1e5e50015d01
using DataFrames

# ╔═╡ 34b2988f-3ae5-4de8-9336-5c250a783e5d
using Distributions

# ╔═╡ e2972627-9b5f-4038-8ba9-621658c06b30
using Turing

# ╔═╡ 2dff25b6-8598-4f26-80e1-17303e7c2c78
md"""
## Generating features
"""

# ╔═╡ f76de121-b119-439e-a1c1-0324587f36a1
@model function generate_person()
    # --- Random intercept ---
    σ_b ~ truncated(Normal(0.0, 1.0), 0.0, Inf)
    b   ~ Normal(0.0, σ_b)

    # --- Multiplicative sensitivities ---
    μ_caff       ~ LogNormal(log(1.15), 0.05)
    σ_caff       ~ truncated(Normal(0.0, 0.08), 0.0, Inf)
    γ_caff_i     ~ LogNormal(log(μ_caff), σ_caff)

    μ_activity   ~ LogNormal(log(1.08), 0.04)
    σ_activity   ~ truncated(Normal(0.0, 0.10), 0.0, Inf)
    γ_activity_i ~ LogNormal(log(μ_activity), σ_activity)

    μ_food       ~ LogNormal(log(1.06), 0.03)
    σ_food       ~ truncated(Normal(0.0, 0.06), 0.0, Inf)
    γ_food_i     ~ LogNormal(log(μ_food), σ_food)

    # --- Stable physiological traits ---
    resting_heart_rate     ~ truncated(Normal(70, 10), 40, 110)
    heart_rate_variability ~ LogNormal(log(35), 0.4)
    sleep_quality          ~ Beta(2, 6)

	# --- Wake time is person dependent ---
	wake_time ~ truncated(Normal(7.0, 1.0), 4.0, 10.0)

    # --- Additive random slopes on stable traits ---
    β_hrv_i     ~ Normal(-0.05, 0.015)
    β_rhr_i     ~ Normal(0.08, 0.02)
    β_sleep_q_i ~ Normal(-3.5, 0.8)

    # --- Person-level slope on sleep duration ---
    β_sleep_dur_i ~ Normal(-1.2, 0.3)

    # --- Observation noise (property of the measurement, not the observation) ---
    σ ~ truncated(Normal(0.0, 0.3), 0.0, Inf)

    person_shift = β_hrv_i     * (heart_rate_variability - 35.0) +
                   β_rhr_i     * (resting_heart_rate     - 70.0) +
                   β_sleep_q_i * (sleep_quality          - 0.25)
	# --- Coefficient diurnal curve (person-dependent) ---
	A  ~ TriangularDist(15, 20, 17.5)


    return (
        b             = b,
        γ_caff_i      = γ_caff_i,
        γ_activity_i  = γ_activity_i,
        γ_food_i      = γ_food_i,
        β_sleep_dur_i = β_sleep_dur_i,
        person_shift  = person_shift,
		wake_time     = wake_time,
		A             = A,
        σ             = σ,
    )
end

# ╔═╡ 5eb714ff-b48c-49b3-a162-a2e1143f0514
@model function generate_observations(person::NamedTuple, n_obs::Int)
    y = Vector{Float64}(undef, n_obs)

    # Declare all observation-level variables as vectors
    t                 ~ filldist(Uniform(0.0, 14.0),             n_obs)
    sleep_duration    ~ filldist(truncated(Normal(7.2, 1.0), 3, 10), n_obs)
    physical_activity ~ filldist(LogNormal(log(6000), 0.5),      n_obs)
    caffeine_intake   ~ filldist(Bernoulli(0.7),                  n_obs)
    food_intake       ~ filldist(Gamma(2, 2),                     n_obs)
    λ                 ~ filldist(Normal(0.3, 0.02),               n_obs)

    for j in 1:n_obs
        base = person.A * exp(-λ[j] * t[j]) + 5.0

        sleep_dur_effect     = person.β_sleep_dur_i * (sleep_duration[j] - 7.2)
        caff_factor          = caffeine_intake[j] == 1 ? person.γ_caff_i : 1.0
        activity_factor      = person.γ_activity_i ^ (log(physical_activity[j]) - log(6000.0))
        food_factor          = person.γ_food_i     ^ (food_intake[j] - 4.0)
        multiplicative_scale = caff_factor * activity_factor * food_factor

        C_nadir  = 5.0
        cortisol = multiplicative_scale * (base - C_nadir) + C_nadir +
                   person.person_shift + sleep_dur_effect + person.b

        y[j] ~ LogNormal(log(max(cortisol, 3.0)), person.σ)
    end

    return y
end

# ╔═╡ a9a608ba-fcc4-4055-a4b7-b56a9fcb673d
function make_patient_data(n_obs::Int = 20)
    person_model  = generate_person()
    person_chain  = sample(person_model, Prior(), 1)
    person_params = generated_quantities(person_model, person_chain) |> only

    obs_model = generate_observations(person_params, n_obs)
    obs_chain = sample(obs_model, Prior(), 1)
    cortisol  = generated_quantities(obs_model, obs_chain) |> only

    # Helper: extract vector of values for a variable stored as "name[1]"..."name[n]"
    function get_vec(chain, name::String, n::Int)
        [chain[Symbol("$name[$j]")][1] for j in 1:n]
    end

    features = DataFrame(
        observation        = 1:n_obs,
        cortisol           = collect(cortisol),
        hours_since_waking = get_vec(obs_chain, "t", n_obs),
        time_of_day        = person_params.wake_time .+ get_vec(obs_chain, "t", n_obs),
        sleep_duration     = get_vec(obs_chain, "sleep_duration", n_obs),
        physical_activity  = get_vec(obs_chain, "physical_activity", n_obs),
        caffeine_intake    = Bool.(get_vec(obs_chain, "caffeine_intake", n_obs)),
        food_intake        = get_vec(obs_chain, "food_intake", n_obs),
    )

    return person_params, features
end

# ╔═╡ d11cbce8-1a90-4806-b8b3-d78402a41d7b
function make_dataset(n_patients::Int, n_obs::Int = 10)
    all_dfs = []
    for i in 1:n_patients
        person, features = make_patient_data(n_obs)
        features.patient_id      .= i
        # Person-level columns — same value repeated for all obs of this person
        features.γ_caff_i        .= person.γ_caff_i
        features.γ_activity_i    .= person.γ_activity_i
        features.γ_food_i        .= person.γ_food_i
        features.β_sleep_dur_i   .= person.β_sleep_dur_i
        features.person_shift    .= person.person_shift
        features.b               .= person.b
        features.σ               .= person.σ
        push!(all_dfs, features)
    end
    return vcat(all_dfs...)
end

# ╔═╡ 5782fa94-a327-4fd6-84cb-fc4af7f740b6
md"""
## Making the dataset
"""

# ╔═╡ 19891b42-8de5-4d93-9a02-bf1615a7fae6
md"""
- 100 donors
- each 20 samples
"""

# ╔═╡ fedf5271-4dd6-4bf0-b088-aec5e926c110
df = make_dataset(100)

# ╔═╡ a6adcb1d-0377-45c1-abac-62de256be97c
begin
    ids = unique(df.patient_id)
    cortisol_groups = [df[df.patient_id .== id, :cortisol] for id in ids]
    
    violin(
        repeat(ids, inner=10),  # x-axis: patient id repeated for each observation
        df.cortisol,
        group  = df.patient_id,
        xlabel = "Patient ID",
        ylabel = "Cortisol (µg/dL)",
        title  = "Cortisol distribution per patient",
        legend = false,
        linewidth = 0.5,
        fillalpha = 0.7,
    )
end

# ╔═╡ 6a871e56-662c-4e5d-ae77-0a09f8bb7729
begin
    physical_groups = [df[df.patient_id .== id, :physical_activity] for id in ids]
    
    violin(
        repeat(ids, inner=10),  # x-axis: patient id repeated for each observation
        df.physical_activity,
        group  = df.patient_id,
        xlabel = "Patient ID",
        ylabel = "Cortisol (µg/dL)",
        title  = "Physical activity per patient",
        legend = false,
        linewidth = 0.5,
        fillalpha = 0.7,
    )
end

# ╔═╡ 059f3bc1-580a-4eb2-a7d5-ebb4e6996119
begin
	# Select only numeric columns and convert to matrix
	numeric_cols = [:cortisol, :hours_since_waking, :time_of_day, 
	                :sleep_duration, :physical_activity, :food_intake]
	
	corrplot(Matrix(df[:, numeric_cols]), 
	         label = string.(numeric_cols),
	         size  = (900, 900))
end

# ╔═╡ aab60acb-dfe6-471c-9d62-c175c1a72a3d
XLSX.writetable("my_data.xlsx", "Sheet1" => df)

# ╔═╡ 2515e9d0-e69a-4c75-ace7-7ce981761d4e
md"""
## Train & Test data
"""

# ╔═╡ 4f0640f2-7a39-4d7d-9884-2ea6a5a58867
names(df)

# ╔═╡ 703b7440-189b-45f3-bc89-1adf401cb854
begin
	X = df[:,[:hours_since_waking, :time_of_day, :sleep_duration, 	:physical_activity, :food_intake, :caffeine_intake]]
	y = df.cortisol
	train_idx, test_idx = partition(eachindex(y), 0.8, shuffle=true)

	X_train = X[train_idx, :]
	X_test = X[test_idx, :]

	y_train = y[train_idx]
	y_test = y[test_idx]
end;

# ╔═╡ 8384f35b-d92f-494d-8178-9169d88516ef
md"""
# Fitting a model
"""

# ╔═╡ 1cbaab8d-a39e-4370-90a3-fda556229b6d
begin
	Booster = @load EvoTreeRegressor # loads code defining a model type
	booster = Booster()   # specify hyper-parameter at construction
end

# ╔═╡ 442db0fc-e5d6-40ac-bbe2-2595a0de9a96
begin
	# Wrap the model and data in a machine:
	mach = machine(booster, X_train, y_train)
	
	# Fit the machine:
	fit!(mach)
	
	# Get training predictions:
	yhat = MLJ.predict(mach, X_test)
end

# ╔═╡ b770e49e-8159-41b5-80bf-e39d2b503eca
evaluate!(
    mach,
    resampling=CV(nfolds=5),
    measures=[rmse],
)

# ╔═╡ 2d5b22a2-ac86-4733-a959-33d4cab28c7d
begin
	scatter(y_test, yhat, xlims = (0, 50), ylims = (0, 50), label = "Prediction")
	plot!(1:1:50, label = "Perfect prediction")
end

# ╔═╡ 0db89ed8-0d44-42ba-a4a5-d3bb9646447c
md"""
## Hyperparametertuning
"""

# ╔═╡ b3cbf83f-4bff-4afa-a1a1-92f97569fc3b
begin
	# Define a hyperparameter range:
	r1 = range(booster, :max_depth, lower=1, upper=10) # values=[2, 3, 4, 6]
	r2 = range(booster, :eta, lower=0.01, upper=0.3)
	
	r_row = range(booster, :rowsample, lower=0.5, upper=1.0)
	r_col = range(booster, :colsample, lower=0.5, upper=1.0)
	
	# Create self-tuning version of model:
	tuned_booster = TunedModel(
	    booster, range=[r1, r2, r_row, r_col], tuning=Grid(resolution = 4), measure=rms, resampling=CV(nfolds=3))
	
	# Train the wrapped model:
	mach_b = machine(tuned_booster, X_train, y_train) |> fit!
	
	# Predict using optimal model:
	yhat_best = MLJ.predict(mach_b, X_test)
	
	# Inspect the best model:
	booster_best = fitted_params(mach_b).best_model
end

# ╔═╡ b10cf869-ea7d-4ee4-b27b-cebaa3ec805e
fitted_params(mach_b).best_model

# ╔═╡ 6dc318c3-4843-4e50-b280-19beca8e767c
plot(mach_b)

# ╔═╡ e6a8f012-4ca3-47bb-af5c-9622f599c44e
entry = report(mach_b).best_history_entry

# ╔═╡ 08e7d824-5b9a-4a86-991c-447eacfa1db5
begin
	entry.model.max_depth
	entry.model.eta
end

# ╔═╡ 1ae64568-bb6d-4408-a720-34bd729259a8
evaluate!(
    mach_b,
    resampling=CV(nfolds=3),
    measures=[rmse],
)

# ╔═╡ 40b98162-b37d-4f64-b217-1727274e2729
begin
	scatter(y_test, yhat_best, xlims = (0, 50), ylims = (0, 50), label = "Prediction", ylabel = "Prediction", xlabel = "True")
	plot!(1:1:50, label = "Perfect prediction")
end

# ╔═╡ Cell order:
# ╠═a93795c0-38e3-11f1-bfa9-8182dfeb44b1
# ╠═05d81ba8-8ba3-44e8-8624-2773438b5944
# ╠═6468d1fa-9186-4267-95ab-dd0b493429ac
# ╠═a6aa8b59-54d8-4879-bce7-1e20f5407b1b
# ╠═8bd1c120-9758-4ae6-bea9-e8bb80e2f64b
# ╠═ebaf341f-398d-4cbf-8bd5-1e5e50015d01
# ╠═34b2988f-3ae5-4de8-9336-5c250a783e5d
# ╠═e2972627-9b5f-4038-8ba9-621658c06b30
# ╟─2dff25b6-8598-4f26-80e1-17303e7c2c78
# ╠═f76de121-b119-439e-a1c1-0324587f36a1
# ╠═5eb714ff-b48c-49b3-a162-a2e1143f0514
# ╠═a9a608ba-fcc4-4055-a4b7-b56a9fcb673d
# ╠═d11cbce8-1a90-4806-b8b3-d78402a41d7b
# ╟─5782fa94-a327-4fd6-84cb-fc4af7f740b6
# ╟─19891b42-8de5-4d93-9a02-bf1615a7fae6
# ╠═fedf5271-4dd6-4bf0-b088-aec5e926c110
# ╠═a6adcb1d-0377-45c1-abac-62de256be97c
# ╠═6a871e56-662c-4e5d-ae77-0a09f8bb7729
# ╠═059f3bc1-580a-4eb2-a7d5-ebb4e6996119
# ╠═aab60acb-dfe6-471c-9d62-c175c1a72a3d
# ╟─2515e9d0-e69a-4c75-ace7-7ce981761d4e
# ╠═4f0640f2-7a39-4d7d-9884-2ea6a5a58867
# ╠═703b7440-189b-45f3-bc89-1adf401cb854
# ╟─8384f35b-d92f-494d-8178-9169d88516ef
# ╠═1cbaab8d-a39e-4370-90a3-fda556229b6d
# ╠═442db0fc-e5d6-40ac-bbe2-2595a0de9a96
# ╠═b770e49e-8159-41b5-80bf-e39d2b503eca
# ╠═2d5b22a2-ac86-4733-a959-33d4cab28c7d
# ╟─0db89ed8-0d44-42ba-a4a5-d3bb9646447c
# ╠═b3cbf83f-4bff-4afa-a1a1-92f97569fc3b
# ╠═b10cf869-ea7d-4ee4-b27b-cebaa3ec805e
# ╠═6dc318c3-4843-4e50-b280-19beca8e767c
# ╠═e6a8f012-4ca3-47bb-af5c-9622f599c44e
# ╠═08e7d824-5b9a-4a86-991c-447eacfa1db5
# ╠═1ae64568-bb6d-4408-a720-34bd729259a8
# ╠═40b98162-b37d-4f64-b217-1727274e2729
