"""Typed identifiers for domain entities.

NewType wrappers over UUID so that a SignalId cannot be passed where a
ProfileId is expected. They are erased at runtime (zero cost) but enforced
by the type checker.
"""

from typing import NewType
from uuid import UUID, uuid4

SignalId = NewType("SignalId", UUID)
ProfileId = NewType("ProfileId", UUID)
CVId = NewType("CVId", UUID)


def new_signal_id() -> SignalId:
    return SignalId(uuid4())


def new_profile_id() -> ProfileId:
    return ProfileId(uuid4())


def new_cv_id() -> CVId:
    return CVId(uuid4())
