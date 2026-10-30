{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

{#- Members are taken from ``mappy_members`` (built in conf.py) so that only
    members defined in mappy_ds are listed; dataclass fields are described by
    the numpydoc "Attributes" section of the class docstring. #}
{%- set info = mappy_members.get(fullname, {}) %}
{%- if info.get("properties") %}

   .. rubric:: Properties

   .. autosummary::
      :toctree:
{% for item in info["properties"] %}
      ~{{ name }}.{{ item }}
{%- endfor %}
{%- endif %}
{%- if info.get("methods") %}

   .. rubric:: Methods

   .. autosummary::
      :toctree:
{% for item in info["methods"] %}
      ~{{ name }}.{{ item }}
{%- endfor %}
{%- endif %}
