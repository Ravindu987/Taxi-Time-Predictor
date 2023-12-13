import React, { useState } from "react";
import {
  TextField,
  Button,
  Grid,
  Card,
  CardContent,
  Typography,
} from "@mui/material";
import { ThemeProvider } from "@mui/material/styles";
import theme from "../theme";
import getResult from "../api/services/getResult";
import "./home.css";

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
  const [result, setResult] = useState(0);

  const handleChange = (e) => {
    const { name, value } = e.target;

    setFormData((prevState) => ({
      ...prevState,
      [name]: value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    console.log(formData);
    const result = await getResult(formData);
    console.log(result);
    setResult(result);
  };

  return (
    <div className="container">
      <ThemeProvider theme={theme}>
        <Grid
          container
          spacing={2}
          alignItems="center"
          justifyContent="center"
          paddingTop={3}
        >
          <Grid item xs={8}>
            <TextField
              fullWidth
              label="Vendor ID"
              type="number"
              name="vendor_id"
              value={formData.vendor_id}
              onChange={handleChange}
            />
          </Grid>
          <Grid item xs={8}>
            <TextField
              fullWidth
              label="Pickup Datetime"
              type="text"
              name="pickup_datetime"
              value={formData.pickup_datetime}
              onChange={handleChange}
            />
          </Grid>
          <Grid item xs={8}>
            <TextField
              fullWidth
              label="Passenger Count"
              type="number"
              name="passenger_count"
              value={formData.passenger_count}
              onChange={handleChange}
            />
          </Grid>
          <Grid item xs={8}>
            <TextField
              fullWidth
              label="Pickup Longitude"
              type="number"
              name="pickup_longitude"
              value={formData.pickup_longitude}
              onChange={handleChange}
            />
          </Grid>
          <Grid item xs={8}>
            <TextField
              fullWidth
              label="Pickup Latitude"
              type="number"
              name="pickup_latitude"
              value={formData.pickup_latitude}
              onChange={handleChange}
            />
          </Grid>
          <Grid item xs={8}>
            <TextField
              fullWidth
              label="Dropoff Longitude"
              type="number"
              name="dropoff_longitude"
              value={formData.dropoff_longitude}
              onChange={handleChange}
            />
          </Grid>
          <Grid item xs={8}>
            <TextField
              fullWidth
              label="Dropoff Latitude"
              type="number"
              name="dropoff_latitude"
              value={formData.dropoff_latitude}
              onChange={handleChange}
            />
          </Grid>
          <Grid item xs={8}>
            <TextField
              fullWidth
              label="Store and Fwd Flag"
              type="text"
              name="store_and_fwd_flag"
              value={formData.store_and_fwd_flag}
              onChange={handleChange}
            />
          </Grid>
          <Grid
            item
            xs={8}
            container
            justifyContent="center"
            style={{ marginTop: "20px", marginBottom: "20px" }}
          >
            <Button
              type="submit"
              variant="contained"
              color="primary"
              onClick={handleSubmit}
              style={{ width: "20%" }}
            >
              Submit
            </Button>
          </Grid>
          <Grid item xs={8} md={6} style={{ marginBottom: "20px" }}>
            <Card>
              <CardContent style={{ padding: "20px", textAlign: "center" }}>
                <Typography variant="body1">
                  Predicted Time: {result}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </ThemeProvider>
    </div>
  );
}
