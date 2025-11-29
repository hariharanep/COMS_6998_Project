from __future__ import annotations

import os
import socket
from datetime import datetime

from typing import Dict, List
from uuid import UUID

from fastapi import FastAPI, HTTPException
from fastapi import Query, Path
from typing import Optional

from models.health import Health
from models.prompt import EnhancePromptReq

import torch
from transformers import pipeline

device = 0 if torch.cuda.is_available() else -1

# Load GPT-2
gpt2_pipe = pipeline("text-generation", model="gpt2", device=device)

port = int(os.environ.get("FASTAPIPORT", 8000))

app = FastAPI(
    title="Backend Service API",
    version="0.0.1",
)

# -----------------------------------------------------------------------------
# Address endpoints
# -----------------------------------------------------------------------------

def make_health(echo: Optional[str], path_echo: Optional[str]=None) -> Health:
    return Health(
        status=200,
        status_message="OK",
        timestamp=datetime.utcnow().isoformat() + "Z",
        ip_address=socket.gethostbyname(socket.gethostname()),
        echo=echo,
        path_echo=path_echo
    )

@app.get("/health", response_model=Health)
def get_health_no_path(echo: str | None = Query(None, description="Optional echo string")):
    # Works because path_echo is optional in the model
    return make_health(echo=echo, path_echo=None)

@app.get("/health/{path_echo}", response_model=Health)
def get_health_with_path(
    path_echo: str = Path(..., description="Required echo in the URL path"),
    echo: str | None = Query(None, description="Optional echo string"),
):
    return make_health(echo=echo, path_echo=path_echo)

@app.post("/enhance")
def enhance(req: EnhancePromptReq):
    enhanced = gpt2_pipe(req.prompt, max_length=256)[0]['generated_text']
    return {"enhanced_prompt": enhanced}


"""
@app.get("/addresses", response_model=List[AddressRead])
def list_addresses(
    street: Optional[str] = Query(None, description="Filter by street"),
    city: Optional[str] = Query(None, description="Filter by city"),
    state: Optional[str] = Query(None, description="Filter by state/region"),
    postal_code: Optional[str] = Query(None, description="Filter by postal code"),
    country: Optional[str] = Query(None, description="Filter by country"),
):
    results = list(addresses.values())

    if street is not None:
        results = [a for a in results if a.street == street]
    if city is not None:
        results = [a for a in results if a.city == city]
    if state is not None:
        results = [a for a in results if a.state == state]
    if postal_code is not None:
        results = [a for a in results if a.postal_code == postal_code]
    if country is not None:
        results = [a for a in results if a.country == country]

    return results

@app.get("/addresses/{address_id}", response_model=AddressRead)
def get_address(address_id: UUID):
    if address_id not in addresses:
        raise HTTPException(status_code=404, detail="Address not found")
    return addresses[address_id]

@app.patch("/addresses/{address_id}", response_model=AddressRead)
def update_address(address_id: UUID, update: AddressUpdate):
    if address_id not in addresses:
        raise HTTPException(status_code=404, detail="Address not found")
    stored = addresses[address_id].model_dump()
    stored.update(update.model_dump(exclude_unset=True))
    addresses[address_id] = AddressRead(**stored)
    addresses[address_id].updated_at = datetime.utcnow().isoformat()
    return addresses[address_id]

# -----------------------------------------------------------------------------
# Person endpoints
# -----------------------------------------------------------------------------
@app.post("/persons", response_model=PersonRead, status_code=201)
def create_person(person: PersonCreate):
    # Each person gets its own UUID; stored as PersonRead
    person_read = PersonRead(**person.model_dump())
    persons[person_read.id] = person_read
    return person_read

@app.get("/persons", response_model=List[PersonRead])
def list_persons(
    uni: Optional[str] = Query(None, description="Filter by Columbia UNI"),
    first_name: Optional[str] = Query(None, description="Filter by first name"),
    last_name: Optional[str] = Query(None, description="Filter by last name"),
    email: Optional[str] = Query(None, description="Filter by email"),
    phone: Optional[str] = Query(None, description="Filter by phone number"),
    birth_date: Optional[str] = Query(None, description="Filter by date of birth (YYYY-MM-DD)"),
    city: Optional[str] = Query(None, description="Filter by city of at least one address"),
    country: Optional[str] = Query(None, description="Filter by country of at least one address"),
):
    results = list(persons.values())

    if uni is not None:
        results = [p for p in results if p.uni == uni]
    if first_name is not None:
        results = [p for p in results if p.first_name == first_name]
    if last_name is not None:
        results = [p for p in results if p.last_name == last_name]
    if email is not None:
        results = [p for p in results if p.email == email]
    if phone is not None:
        results = [p for p in results if p.phone == phone]
    if birth_date is not None:
        results = [p for p in results if str(p.birth_date) == birth_date]

    # nested address filtering
    if city is not None:
        results = [p for p in results if any(addr.city == city for addr in p.addresses)]
    if country is not None:
        results = [p for p in results if any(addr.country == country for addr in p.addresses)]

    return results

@app.get("/persons/{person_id}", response_model=PersonRead)
def get_person(person_id: UUID):
    if person_id not in persons:
        raise HTTPException(status_code=404, detail="Person not found")
    return persons[person_id]

@app.patch("/persons/{person_id}", response_model=PersonRead)
def update_person(person_id: UUID, update: PersonUpdate):
    if person_id not in persons:
        raise HTTPException(status_code=404, detail="Person not found")
    stored = persons[person_id].model_dump()
    stored.update(update.model_dump(exclude_unset=True))
    persons[person_id] = PersonRead(**stored)
    persons[person_id].updated_at = datetime.utcnow().isoformat()
    return persons[person_id]

# -----------------------------------------------------------------------------
# Organization endpoints
# -----------------------------------------------------------------------------
@app.post("/organization", response_model=OrganizationRead, status_code=201)
def create_organization(organization: OrganizationCreate):
    if organization.id in organizations:
        raise HTTPException(status_code=400, detail="Organization with this ID already exists")
    organization_read = OrganizationRead(**organization.model_dump())
    organizations[organization_read.id] = organization_read
    return organization_read

@app.get("/organizations", response_model=List[OrganizationRead])
def list_organizations(
        org_name: Optional[str] = Query(None, description="Filter by organization name"),
        email: Optional[str] = Query(None, description="Filter by email"),
        phone: Optional[str] = Query(None, description="Filter by phone number"),
        city: Optional[str] = Query(None, description="Filter by city of address"),
        country: Optional[str] = Query(None, description="Filter by country of address"),
):
    results = list(organizations.values())

    if org_name is not None:
        results = [p for p in results if p.org_name == org_name]
    if email is not None:
        results = [p for p in results if p.email == email]
    if phone is not None:
        results = [p for p in results if p.phone == phone]

    # nested address filtering
    if city is not None:
        results = [p for p in results if p.address.city == city]
    if country is not None:
        results = [p for p in results if p.address.country == country]

    return results

@app.get("/organizations/{organization_id}", response_model=OrganizationRead)
def get_organization(organization_id: UUID):
    if organization_id not in organizations:
        raise HTTPException(status_code=404, detail="Organization not found")
    return organizations[organization_id]

@app.patch("/organizations/{organization_id}", response_model=OrganizationRead)
def update_organization(organization_id: UUID, update: OrganizationUpdate):
    if organization_id not in organizations:
        raise HTTPException(status_code=404, detail="Organization not found")
    stored = organizations[organization_id].model_dump()
    stored.update(update.model_dump(exclude_unset=True))
    organizations[organization_id] = OrganizationRead(**stored)
    organizations[organization_id].updated_at = datetime.utcnow().isoformat()
    return organizations[organization_id]


# -----------------------------------------------------------------------------
# House endpoints
# -----------------------------------------------------------------------------
@app.post("/house", response_model=HouseRead, status_code=201)
def create_house(house: HouseCreate):
    if house.id in houses:
        raise HTTPException(status_code=400, detail="House with this ID already exists")
    house_read = HouseRead(**house.model_dump())
    houses[house_read.id] = house_read
    return house_read

@app.get("/houses", response_model=List[HouseRead])
def list_houses(
        phone: Optional[str] = Query(None, description="Filter by phone number"),
        person_uni: Optional[UNIType] = Query(None, description="Filter by person"),
        city: Optional[str] = Query(None, description="Filter by city of address"),
        country: Optional[str] = Query(None, description="Filter by country of address"),
):
    results = list(houses.values())

    if phone is not None:
        results = [p for p in results if p.phone == phone]

    # nested address filtering
    if city is not None:
        results = [p for p in results if p.address.city == city]
    if country is not None:
        results = [p for p in results if p.address.country == country]

    # nested person filtering
    if person_uni is not None:
        results = [p for p in results if any(uni == person_uni for uni in p.people)]

    return results

@app.get("/houses/{house_id}", response_model=HouseRead)
def get_house(house_id: UUID):
    if house_id not in houses:
        raise HTTPException(status_code=404, detail="House not found")
    return houses[house_id]

@app.patch("/houses/{house_id}", response_model=HouseRead)
def update_house(house_id: UUID, update: HouseUpdate):
    if house_id not in houses:
        raise HTTPException(status_code=404, detail="House not found")
    stored = houses[house_id].model_dump()
    stored.update(update.model_dump(exclude_unset=True))
    houses[house_id] = HouseRead(**stored)
    houses[house_id].updated_at = datetime.utcnow().isoformat()
    return houses[house_id]

"""

# -----------------------------------------------------------------------------
# Root
# -----------------------------------------------------------------------------
@app.get("/")
def root():
    return {"message": "Welcome to the Backend Service API."}

# -----------------------------------------------------------------------------
# Entrypoint for `python main.py`
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
