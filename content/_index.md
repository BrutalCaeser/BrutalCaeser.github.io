---
# Leave the homepage title empty to use the site title
title: ''
date: 2022-10-24
type: landing

design:
  # Default section spacing
  spacing: '6rem'

sections:
  - block: resume-biography-3
    content:
      username: admin
      text: ''
      button:
        text: Download CV
        url: uploads/resume_yashvardhan_gupta.pdf
      headings:
        about: ''
        education: ''
        interests: ''
    design:
      css_class: hbx-bg-gradient
      avatar:
        size: large
        shape: circle

  - block: collection
    id: projects
    content:
      title: Projects
      text: Built from scratch — models, experiments, and tools. Every one taught me something I didn't know to look for.
      filters:
        folders:
          - projects
      count: 0
      sort_by: 'Date'
      sort_ascending: false
    design:
      view: article-grid
      fill_image: false
      columns: 3

  - block: collection
    id: papers
    content:
      title: Publications
      filters:
        folders:
          - publications
        featured_only: true
    design:
      view: masonry
      columns: 2

  - block: collection
    id: blog
    content:
      title: Recent Writing
      text: Deep dives into the math behind ML.
      filters:
        folders:
          - blog
      count: 3
      sort_by: 'Date'
      sort_ascending: false
    design:
      view: article-grid
      columns: 3
---
