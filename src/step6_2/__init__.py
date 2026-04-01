"""Step 6_2 package.

Keep package import lightweight so cluster submit wrappers can import
`src.step6_2.config` on login nodes without requiring the full training stack.
Import concrete functionality from submodules directly.
"""

__all__: list[str] = []
