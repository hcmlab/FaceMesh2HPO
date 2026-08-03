import {useEffect, useRef} from "react";

export const DistributionChart = ({title, data, color, type = 'bar', centerText}: {
    title?: string,
    data: { [key: string]: number },
    color: string,
    type?: 'bar' | 'pie',
    centerText?: string
}) => {
    const chartRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (!chartRef.current || Object.keys(data).length === 0) return;

        const labels = Object.keys(data);
        const values = Object.values(data);

        let plotData: any[] = [];
        const layout: any = {
            margin: {t: 20, b: 20, l: 40, r: 10},
            height: 250, // Increased size
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            showlegend: false,
            font: {size: 9}
        };
        if (title) {
            layout["title"] = {
                text: title,
                font: {size: 10, color: '#666'}
            };
        }

        if (type === 'pie') {
            plotData = [{
                values,
                labels,
                type: 'pie',
                textinfo: 'label+percent',
                textposition: 'inside',
                hole: 0.4,
                marker: {
                    colors: labels.map((_, i) => `${color}${Math.max(20, 100 - i * 20)}`)
                }
            }];
            layout.showlegend = false;
            layout.margin = {t: 10, b: 10, l: 10, r: 10};

            // CENTER TEXT ANNOTATION
            layout.annotations = [{
                text: centerText || values.reduce((sum, v) => sum + v, 0),  // Default: total count
                showarrow: false,
                xref: 'paper',
                yref: 'paper',
                x: 0.5,    // Center X
                y: 0.5,    // Center Y
                font: {
                    size: 10,
                    color: '#495057',
                    family: 'Arial Black, sans-serif'
                },
            }];
        } else {
            plotData = [{
                x: labels,
                y: values,
                type: 'bar',
                marker: {
                    color: color
                }
            }];
            layout.xaxis = {tickfont: {size: 8}};
            layout.yaxis = {tickfont: {size: 8}, gridcolor: '#eee'};
        }

        const config = {
            responsive: true,
            displayModeBar: false
        };

        window.Plotly.newPlot(chartRef.current, plotData, layout, config);

        return () => {
            if (chartRef.current) {
                window.Plotly.purge(chartRef.current);
            }
        };
    }, [data, title, color, type]);

    return <div ref={chartRef} className="w-100" style={{minHeight: '250px'}}/>; // Increased size
};