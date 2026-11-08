using MappyDS
using Documenter
using DocumenterCitations
using DocumenterInterLinks

DocMeta.setdocmeta!(MappyDS, :DocTestSetup, :(using MappyDS); recursive = true)

# Cross-references to the Python documentation (Sphinx objects.inv), e.g.
# [`PoincareMap`](@extref mappy_ds mappy_ds.PoincareMap).
links = InterLinks(
    "mappy_ds" => "https://yuu-miino.github.io/doc.mappy/latest/objects.inv",
)
bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"); style = :numeric)

makedocs(;
    modules = [MappyDS],
    authors = "Yuu Miino",
    sitename = "MappyDS.jl",
    format = Documenter.HTML(;
        canonical = "https://yuu-miino.github.io/doc.mappy/julia/stable/",
        edit_link = "main",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "Python package" => "python.md",
        "References" => "references.md",
    ],
    plugins = [links, bib],
)

# The docs live in the shared documentation repository `doc.mappy`, next to
# the Sphinx docs of the Python package, under the subdirectory `julia/`.
# Pushing to another repository requires the SSH deploy key DOCUMENTER_KEY
# (GITHUB_TOKEN only reaches the repository the workflow runs in).
deploydocs(;
    repo = "github.com/Yuu-Miino/MappyDS.jl.git",
    deploy_repo = "github.com/Yuu-Miino/doc.mappy.git",
    branch = "gh-pages",
    dirname = "julia",
    devbranch = "main",
    devurl = "dev",
    versions = ["stable" => "v^", "v#.#", "dev" => "dev"],
    push_preview = false,
)
