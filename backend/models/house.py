from __future__ import annotations

from typing import Optional, List, Annotated
from uuid import UUID, uuid4
from datetime import date, datetime
from pydantic import BaseModel, Field

from .address import AddressBase
from .person import UNIType



class HouseBase(BaseModel):
    id: UUID = Field(
        default_factory=uuid4,
        description="Persistent House ID (server-generated).",
        json_schema_extra={"example": "550e8400-e29b-41d4-a716-446655440000"},
    )
    phone: Optional[str] = Field(
        None,
        description="Contact phone number in any reasonable format.",
        json_schema_extra={"example": "+1-212-555-0199"},
    )

    # Embed address (each with persistent ID)
    address: AddressBase = Field(
        ...,
        description="Address linked to this house (carries a persistent Address ID).",
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
    ),
    people: List[UNIType] = Field(
        default_factory=list,
        description="People living inside this house.",
        json_schema_extra={
            "example": [
                "abc1234",
                "zde3421"
            ]
        }
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": "12345678-e29b-41d4-a716-123456781234",
                    "phone": "+1-212-555-0199",
                    "address": {
                        "id": "550e8400-e29b-41d4-a716-446655440000",
                        "street": "123 Main St",
                        "city": "London",
                        "state": None,
                        "postal_code": "SW1A 1AA",
                        "country": "UK",
                    },
                    "people": [
                        "abc1234",
                        "zde3421"
                    ]
                }
            ]
        }
    }


class HouseCreate(HouseBase):
    """Creation payload for a House."""
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": "12345678-e29b-41d4-a716-123456781234",
                    "phone": "+1-212-555-0199",
                    "address": {
                        "id": "550e8400-e29b-41d4-a716-446655440000",
                        "street": "123 Main St",
                        "city": "London",
                        "state": None,
                        "postal_code": "SW1A 1AA",
                        "country": "UK",
                    },
                    "people": [
                        "abc1234",
                        "zde3421"
                    ]
                }
            ]
        }
    }


class HouseUpdate(BaseModel):
    """Partial update for a House; supply only fields to change."""
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
    people: Optional[List[UUID]] = Field(
        None,
        description="Replace the entire set of people with this list.",
        json_schema_extra={
            "example": [
                "abc1234",
                "zde3421"
            ]
        },
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
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


class HouseRead(HouseBase):
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
                    "phone": "+1-212-555-0199",
                    "address": {
                        "id": "550e8400-e29b-41d4-a716-446655440000",
                        "street": "123 Main St",
                        "city": "London",
                        "state": None,
                        "postal_code": "SW1A 1AA",
                        "country": "UK",
                    },
                    "people": [
                        "abc1234",
                        "zde3421"
                    ],
                    "created_at": "2025-01-15T10:20:30Z",
                    "updated_at": "2025-01-16T12:00:00Z"
                }
            ]
        }
    }
