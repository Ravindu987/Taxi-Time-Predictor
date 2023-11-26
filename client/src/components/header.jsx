import React from "react";
import { Box } from "@mui/system";
import { Typography } from "@mui/material";
import { ThemeProvider } from "@mui/material/styles";
import Grid from "@mui/material/Grid";
import theme from "../theme";

export function Header() {
  return (
    <ThemeProvider theme={theme}>
      <Grid
        container
        direction="row"
        flex={1}
        justifyContent="space-evenly"
        alignItems="center"
        sx={{ textAlign: "center" }}
      >
        <Grid item xs={12}>
          <Box bgcolor="primary.main" p={2} sx={{ textAlign: "center" }}>
            <Typography variant="h4" gutterBottom m={0}>
              Taxi Time Predictor
            </Typography>
          </Box>
        </Grid>
      </Grid>
    </ThemeProvider>
  );
}

export default Header;
