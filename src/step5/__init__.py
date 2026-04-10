"""Step 5 package.

Keep package import lightweight so cluster submit wrappers can import
`src.step5.config` on login nodes without requiring the full training stack.
Import concrete functionality from submodules directly.
"""

__all__: list[str] = []
