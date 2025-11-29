from __future__ import annotations

from typing import Optional, List, Annotated
from uuid import UUID, uuid4
from datetime import date, datetime
from pydantic import BaseModel, Field, EmailStr

from .address import AddressBase


class OrganizationBase(BaseModel):
    id: UUID = Field(
        default_factory=uuid4,
        description="Persistent Organization ID (server-generated).",
        json_schema_extra={"example": "550e8400-e29b-41d4-a716-446655440000"},
    )
    org_name: str = Field(
        ...,
        description="Given organization name.",
        json_schema_extra={"example": "Computer Science"},
    )
    email: EmailStr = Field(
        ...,
        description="Primary email address.",
        json_schema_extra={"example": "columbia_cs@columbia.edu"},
    )
    phone: Optional[str] = Field(
        None,
        description="Contact phone number in any reasonable format.",
        json_schema_extra={"example": "+1-212-555-0199"},
    )

    # Embed address (each with persistent ID)
    address: AddressBase = Field(
        ...,
        description="Address linked to this organization (carries a persistent Address ID).",
        json_schema_extra={
            "example":
                {
                    "id": "550e8400-e29b-41d4-a716-446655440000",
                    "street": "123 Main St",
                    "city": "London",
                    "state": None,
                    "postal_code": "SW1A 1AA",
                    "country": "UK",
                }
        },
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": "12345678-e29b-41d4-a716-123456781234",
                    "org_name": "Computer Science",
                    "email": "columbia_cs@columbia.edu",
                    "phone": "+1-212-555-0199",
                    "address": {
                        "id": "550e8400-e29b-41d4-a716-446655440000",
                        "street": "123 Main St",
                        "city": "London",
                        "state": None,
                        "postal_code": "SW1A 1AA",
                        "country": "UK",
                    }
                }
            ]
        }
    }


class OrganizationCreate(OrganizationBase):
    """Creation payload for an Organization."""
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": "12345678-e29b-41d4-a716-123456781234",
                    "org_name": "Computer Science",
                    "email": "columbia_cs@columbia.edu",
                    "phone": "+1-212-555-0199",
                    "address": {
                        "id": "550e8400-e29b-41d4-a716-446655440000",
                        "street": "123 Main St",
                        "city": "New York",
                        "state": "NY",
                        "postal_code": "10025",
                        "country": "US",
                    }
                }
            ]
        }
    }


class OrganizationUpdate(BaseModel):
    """Partial update for an Organization; supply only fields to change."""
    org_name: Optional[str] = Field(None, json_schema_extra={"example": "Computer Science"})
    email: Optional[EmailStr] = Field(None, json_schema_extra={"example": "columbia_cs@columbia.edu"})
    phone: Optional[str] = Field(None, json_schema_extra={"example": "+1-212-555-0199"})
    address: Optional[AddressBase] = Field(
        None,
        description="Replace the old address with this new address.",
        json_schema_extra={
            "example":
                {
                    "id": "550e8400-e29b-41d4-a716-446655440000",
                    "street": "123 Main St",
                    "city": "New York",
                    "state": "NY",
                    "postal_code": "10025",
                    "country": "US",
                }
        }
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"org_name": "Math", "email": "math@columbia.edu"},
                {"phone": "+1-415-555-0199"},
                {
                    "address": {
                            "id": "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
                            "street": "10 Downing St",
                            "city": "London",
                            "state": None,
                            "postal_code": "SW1A 2AA",
                            "country": "UK",
                    }

                }
            ]
        }
    }


class OrganizationRead(OrganizationBase):
    """Server representation returned to clients."""
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Creation timestamp (UTC).",
        json_schema_extra={"example": "2025-01-15T10:20:30Z"},
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Last update timestamp (UTC).",
        json_schema_extra={"example": "2025-01-16T12:00:00Z"},
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": "12345678-e29b-41d4-a716-123456781234",
                    "org_name": "Computer Science",
                    "email": "columbia_cs@columbia.edu",
                    "phone": "+1-212-555-0199",
                    "address": {
                        "id": "550e8400-e29b-41d4-a716-446655440000",
                        "street": "123 Main St",
                        "city": "London",
                        "state": None,
                        "postal_code": "SW1A 1AA",
                        "country": "UK",
                    },
                    "created_at": "2025-01-15T10:20:30Z",
                    "updated_at": "2025-01-16T12:00:00Z"
                }
            ]
        }
    }
