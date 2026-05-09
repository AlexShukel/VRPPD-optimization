export interface Location {
    hash: string;
    latitude: number;
    longitude: number;
}

export interface Order {
    id: number;
    pickupLocation: Location;
    deliveryLocation: Location;
    loadFactor: number;
}

export interface Vehicle {
    id: number;
    startLocation: Location;
    priceKm: number;
}

export interface Problem {
    vehicles: Vehicle[];
    orders: Order[];
}
