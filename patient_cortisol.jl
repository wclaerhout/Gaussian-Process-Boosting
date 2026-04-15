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
## Generating data
"""

# ╔═╡ f76de121-b119-439e-a1c1-0324587f36a1
@model function generate_data(n)
	time_of_day ~ Uniform(0, 24)
	heart_rate_variability ~ LogNormal(log(35), 0.4)
	resting_heart_rate ~ truncated(Normal(70, 10), 40, 110)
	sleep_duration ~ truncated(Normal(7.2, 1.0), 3, 10)
	sleep_quality ~ Beta(2, 6)
	physical_activity ~ LogNormal(log(6000), 0.5) # number of steps
	caffeine_intake ~ Bernoulli(0.7) # binary
	food_intake ~ Gamma(2, 2) # Time since last meal
	person ~ missing # is the grouping variable
end

# ╔═╡ Cell order:
# ╠═a93795c0-38e3-11f1-bfa9-8182dfeb44b1
# ╠═a6aa8b59-54d8-4879-bce7-1e20f5407b1b
# ╠═ebaf341f-398d-4cbf-8bd5-1e5e50015d01
# ╠═34b2988f-3ae5-4de8-9336-5c250a783e5d
# ╠═e2972627-9b5f-4038-8ba9-621658c06b30
# ╟─2dff25b6-8598-4f26-80e1-17303e7c2c78
# ╠═f76de121-b119-439e-a1c1-0324587f36a1
