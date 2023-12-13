import { createTheme } from "@mui/material";

const theme = createTheme({
    palette: {
        primary: {
            main: "#dc9700",
        },
        secondary: {
            main: "#ffc61a",
        },
        background: {
            default: "#d9d9d9",
            paper: '#efcf96', // your color
        },
    },
});

export default theme;