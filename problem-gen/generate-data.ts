/**
 * @file generate-data.ts
 * @description
 * This script serves as the data seeding utility.
 *
 * Purposes:
 * 1. Ingests raw CSV data representing real-world shipping lanes.
 * 2. Transforms CSV rows into `Order` objects, calculating geohashes and prices.
 * 3. Synthetically generates a large fleet of `Vehicle` objects scattered near pickup locations.
 * 4. Persists the normalized data into `data/orders_<ts>.json` and `data/vehicles_<ts>.json`.
 */

import csv from 'csv-parser';
import fs from 'fs';
import z from 'zod';
import Geohash from 'latlon-geohash';
import path from 'path';

import type { Order, Vehicle } from './src/types';

const getRandomFloat = (min: number, max: number) => {
    return Math.random() * (max - min) + min;
};

const getRandomPriceKm = () => {
    return getRandomFloat(1, 3);
};

// Earth radius (km)
const EARTH_RADIUS_KM = 6371;

const getRandomCoordsInRadius = (centerLat: number, centerLon: number, radiusKm: number) => {
    const lat1 = centerLat * (Math.PI / 180);
    const lon1 = centerLon * (Math.PI / 180);

    const maxDistRad = radiusKm / EARTH_RADIUS_KM;

    const distance = Math.acos(Math.random() * (Math.cos(maxDistRad) - 1) + 1);
    const bearing = 2 * Math.PI * Math.random();

    const lat2 = Math.asin(
        Math.sin(lat1) * Math.cos(distance) + Math.cos(lat1) * Math.sin(distance) * Math.cos(bearing),
    );

    const lon2 =
        lon1 +
        Math.atan2(
            Math.sin(bearing) * Math.sin(distance) * Math.cos(lat1),
            Math.cos(distance) - Math.sin(lat1) * Math.sin(lat2),
        );

    return {
        latitude: lat2 * (180 / Math.PI),
        longitude: lon2 * (180 / Math.PI),
    };
};

const seedDatasetSchema = z
    .object({
        ID: z.string().transform(val => Number.parseInt(val)),
        MODEL_LF: z.string().transform(val => Number.parseInt(val)),
        MODEL_NAME: z.string(),
        CRG_LOAD_LOC_ID: z.string(),
        COUNTRY_FROM: z.string(),
        CITY_FROM: z.string(),
        LAT_FROM: z.string().transform(val => Number.parseFloat(val)),
        LON_FROM: z.string().transform(val => Number.parseFloat(val)),
        CRG_DELIVERY_LOC_ID: z.string(),
        COUNTRY_TO: z.string(),
        CITY_TO: z.string(),
        LAT_TO: z.string().transform(val => Number.parseFloat(val)),
        LON_TO: z.string().transform(val => Number.parseFloat(val)),
        VEHICLE_QNTY: z.string().transform(val => Number.parseInt(val)),
        DISTANCE_KM: z.string().transform(val => Number.parseInt(val)),
    })
    .array();

const GEOHASH_PRECISION = 6;

const generateOrders = (seedDataset: z.infer<typeof seedDatasetSchema>): Order[] => {
    const orders: Order[] = [];

    for (const { ID, LAT_FROM, LON_FROM, LAT_TO, LON_TO, MODEL_LF } of seedDataset) {
        orders.push({
            id: ID,
            pickupLocation: {
                hash: Geohash.encode(LAT_FROM, LON_FROM, GEOHASH_PRECISION),
                latitude: LAT_FROM,
                longitude: LON_FROM,
            },
            deliveryLocation: {
                hash: Geohash.encode(LAT_TO, LON_TO, GEOHASH_PRECISION),
                latitude: LAT_TO,
                longitude: LON_TO,
            },
            loadFactor: MODEL_LF,
            // price: DISTANCE_KM * getRandomPriceKm(),
        } satisfies Order);
    }

    return orders;
};

const VEHICLES_N = 10000;
const VEHICLE_SPAWN_RADIUS_KM = 200;

const generateVehicles = (orders: Order[]): Vehicle[] => {
    if (orders.length === 0) {
        throw new Error('No orders found to generate vehicle locations.');
    }

    const vehicles: Vehicle[] = new Array(VEHICLES_N);

    for (let i = 0; i < VEHICLES_N; ++i) {
        const randomOrder = orders[Math.floor(Math.random() * orders.length)];
        const centerLat = randomOrder.pickupLocation.latitude;
        const centerLon = randomOrder.pickupLocation.longitude;

        const { latitude, longitude } = getRandomCoordsInRadius(centerLat, centerLon, VEHICLE_SPAWN_RADIUS_KM);

        vehicles[i] = {
            id: i + 1,
            priceKm: getRandomPriceKm(),
            startLocation: {
                hash: Geohash.encode(latitude, longitude, GEOHASH_PRECISION),
                latitude,
                longitude,
            },
        };
    }

    return vehicles;
};

const main = async () => {
    const raw: Record<string, string>[] = [];

    fs.createReadStream('seed-dataset.csv')
        .pipe(csv())
        .on('data', data => raw.push(data))
        .on('end', () => {
            const orders = generateOrders(seedDatasetSchema.parse(raw));

            const vehicles = generateVehicles(orders);

            const timestamp = new Date().getTime();
            const dataDir = path.resolve(__dirname, 'data');

            if (!fs.existsSync(dataDir)) {
                fs.mkdirSync(dataDir);
            }

            fs.writeFileSync(path.resolve(dataDir, `orders_${timestamp}.json`), JSON.stringify(orders));
            fs.writeFileSync(path.resolve(dataDir, `vehicles_${timestamp}.json`), JSON.stringify(vehicles));
        });
};

main().catch(err => {
    console.error(err);
    process.exit(1);
});
