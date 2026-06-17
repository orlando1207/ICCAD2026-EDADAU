"""Vendored subset of the analytic legalizer (constraints / skyline / quadratic /
topology), copied into CNN_RL so this package has ZERO dependency on the external
`analytic_legalizer/` directory (avoids merge conflicts). These are pure functions;
the copies are byte-for-byte from analytic_legalizer at vendor time and keep their
own internal `from .constraints import ...` relative imports."""
