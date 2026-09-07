from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.api.deps import get_session
from app.api.errors import Conflict, NotFound
from app.api.schemas import ProfileCreate, ProfileCreated, ProfileDetail, ProfileStats
from app.db.repository import ProfileRepository

router = APIRouter(prefix="/api/v1/profiles", tags=["profiles"])


@router.post("", status_code=status.HTTP_201_CREATED, response_model=ProfileCreated)
def create_profile(payload: ProfileCreate, session: Session = Depends(get_session)):
    repo = ProfileRepository(session)
    if repo.find_by_domain(payload.domain) is not None:
        raise Conflict(
            "a profile already exists for this domain",
            {"domain": payload.domain},
        )
    profile = repo.create(
        name=payload.name,
        domain=payload.domain,
        industry=payload.industry,
        description=payload.description,
        competitors=payload.competitors,
    )
    return ProfileCreated(
        profile_uuid=profile.profile_uuid,
        name=profile.name,
        domain=profile.domain,
        created_at=profile.created_at,
    )


@router.get("/{profile_uuid}", response_model=ProfileDetail)
def get_profile(profile_uuid: str, session: Session = Depends(get_session)):
    repo = ProfileRepository(session)
    profile = repo.get(profile_uuid)
    if profile is None:
        raise NotFound("profile not found", {"profile_uuid": profile_uuid})
    return ProfileDetail(
        profile_uuid=profile.profile_uuid,
        name=profile.name,
        domain=profile.domain,
        industry=profile.industry,
        description=profile.description,
        competitors=list(profile.competitors or []),
        created_at=profile.created_at,
        stats=ProfileStats(**repo.summary_stats(profile_uuid)),
    )
