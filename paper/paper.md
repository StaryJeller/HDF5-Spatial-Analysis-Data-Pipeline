---
title: 'My Research Software Title'
authors:
  - name: Your Name
    affiliation: '1'
affiliations:
  - name: Your Institution
    index: 1
---

# Summary

Provide a concise summary of the software and its purpose for a broad, non-specialist audience. Explain what the software does and why it matters in a research context.

Raw beamline data ->
Extract scientific metadata ->
Resolve acquisition geometry ->
Construct canonical spatial-spectral dataset ->
Export to interchange format(s)

# Statement of need

Describe the problem that the software addresses and why it is important. Explain who the intended users are and how the software fits into the broader research landscape.

This software intends to address a data preservation and interoperability issue from raw binary metadata and provide a functional data pipeline to parse HDF5 binary attributes, populate an intermediary interchange array, and provide options for data export to preserve underlying spectral information in a specified output. This software would aim to bridge a data interpretation gap between beamline sciences and downstream data analysis.

# State of the field

Briefly describe related software or methods in the field and explain how your work compares to or builds on them. If relevant, explain what gap your project addresses.

Current methods in the field, at least from what I've seen, seem to revolve around very lightweight visual exports and post-process data analysis through simple extraction imagery analysis. By amending this process with a comprehensive metadata preservation parsing toolkit, my expectation is to have a verifiable, archival-quality dataset ready for export into various analytical software tools for extensive spatial data analysis.

# Software design

Describe the main design choices, architecture, or workflow decisions that shape the software. Focus on meaningful trade-offs and why the chosen approach is appropriate for the research problem.

Root_folder/
  readers/
    cls.py
    aps.py
    generic.py
  metadata/
    extract.py
    validate.py
  geometry/
    scan-pattern.py
    raster.py
    reconstruct.py
  dataset/
    canonical.py
      data
      coordinates
        x
        y
      measurements
        intensity
      metadata
        beamline
        detector
        scan-pattern
  exporters/
    netcdf.py
    geotiff.py
    zarr.py

# Research impact

Summarize the current or expected impact of the software. This can include reproducible analyses, adoption by others, or evidence of significance in the research workflow. Keep this specific and realistic rather than promotional.

# Project evolution and lessons learned

Use this section to draft short reflections from each in-class activity. These notes can later be refined into the final report and should connect the course activities to your project.

## Modeling Intro

Draft a short reflection on what you learned in the modeling intro activities, which tools or techniques were most useful, how the work relates to your project, and whether you plan to adopt any of the ideas.

## Analytical Modeling

Draft a short reflection on what you learned in the analytical modeling activities, which methods or tools were most useful, how the work relates to your project, and whether you plan to adopt any of the ideas.

## Physical Modeling

Draft a short reflection on what you learned in the physical modeling activities, which methods or tools were most useful, how the work relates to your project, and whether you plan to adopt any of the ideas.

## Data-Driven Modeling

Draft a short reflection on what you learned in the data-driven modeling activities, which methods or tools were most useful, how the work relates to your project, and whether you plan to adopt any of the ideas.

# AI usage disclosure

Describe whether generative AI tools were used in the development of the software, documentation, or manuscript. If no AI tools were used, state that clearly. If they were used, describe the nature of the assistance and how the results were checked.

# Acknowledgements

List any contributors, funding sources, or support that should be acknowledged.

# References

Add references to related software, methods, and relevant literature.
