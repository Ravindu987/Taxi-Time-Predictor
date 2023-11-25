import React, { useState } from "react";
import { TextField, Button, Container, Typography, Grid } from "@mui/material";
import { ThemeProvider } from "@mui/material/styles";
import theme from "../theme";

export function Home() {
  const [formData, setFormData] = useState({
    vendorID: 0,
    pickup_datetime: "",
    passenger_count: 0,
    pickup_longitude: 0,
    pickup_latitude: 0,
    dropoff_longitude: 0,
    dropoff_latitude: 0,
    store_and_fwd_flag: "",
  });

  const handleChange = (e) => {
    const { name, value } = e.target;

    setFormData((prevState) => ({
      ...prevState,
      [name]: value,
    }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    console.log(formData);
  };

  return (
    <ThemeProvider theme={theme}>
      <Container>
        <Typography variant="h4" gutterBottom color="primary">
          Taxi Form
        </Typography>
        <form onSubmit={handleSubmit}>
          <Grid container spacing={2}>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Vendor ID"
                type="number"
                name="vendor_id"
                value={formData.vendor_id}
                onChange={handleChange}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Pickup Datetime"
                type="text"
                name="pickup_datetime"
                value={formData.pickup_datetime}
                onChange={handleChange}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Passenger Count"
                type="number"
                name="passenger_count"
                value={formData.passenger_count}
                onChange={handleChange}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Pickup Longitude"
                type="number"
                name="pickup_longitude"
                value={formData.pickup_longitude}
                onChange={handleChange}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Pickup Latitude"
                type="number"
                name="pickup_latitude"
                value={formData.pickup_latitude}
                onChange={handleChange}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Dropoff Longitude"
                type="number"
                name="dropoff_longitude"
                value={formData.dropoff_longitude}
                onChange={handleChange}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Dropoff Latitude"
                type="number"
                name="dropoff_latitude"
                value={formData.dropoff_latitude}
                onChange={handleChange}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Store and Fwd Flag"
                type="text"
                name="store_and_fwd_flag"
                value={formData.store_and_fwd_flag}
                onChange={handleChange}
              />
            </Grid>
          </Grid>
          <Button type="submit" variant="contained" color="primary">
            Submit
          </Button>
        </form>
      </Container>
    </ThemeProvider>
  );
}
