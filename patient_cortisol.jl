### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ a93795c0-38e3-11f1-bfa9-8182dfeb44b1
using Pkg; Pkg.activate(".")

# ╔═╡ a6aa8b59-54d8-4879-bce7-1e20f5407b1b
using StatsPlots, PlutoUI; TableOfContents()

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
function make_patient_data(n_obs::Int = 10)
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
- each 10 samples
"""

# ╔═╡ fedf5271-4dd6-4bf0-b088-aec5e926c110
df = make_dataset(100)

# ╔═╡ a6adcb1d-0377-45c1-abac-62de256be97c
df[1, :]

# ╔═╡ Cell order:
# ╠═a93795c0-38e3-11f1-bfa9-8182dfeb44b1
# ╠═a6aa8b59-54d8-4879-bce7-1e20f5407b1b
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
