import React, { Component } from "react";
import Graph from "./graph";
import ModelAPI from "./modelapi";
import "./App.css";

import { Form, Select, InputNumber, Button } from "antd";

const { Option } = Select;

class Covid19App extends Component {
  onFinish = values => {
    const { countries, weeks } = values;
    this.setState({
      countries: countries,
      weeks: weeks
    });
  };

  constructor() {
    super();

    this.modelAPI = new ModelAPI();

    this.state = {
      // Placeholder values, should be loaded from some data source eventually.
      countries: null,
      countriesList: ["United States of America", "China"],
      weeks: null
    };
  }

  render() {
    const { countries, countriesList, weeks } = this.state;

    const countryOptions = countriesList.map(c => (
      <Option key={c}> {c} </Option>
    ));

    const mainGraphData = {
      foo: "bar"
    };

    return (
      <div className="covid-19-app">
        <div className="form">
          <Form onFinish={this.onFinish}>
            <Form.Item
              label="Countries"
              name="countries"
              rules={[{ required: true, message: "Please select countries!" }]}
            >
              <Select
                mode="multiple"
                style={{ width: "100%" }}
                placeholder="Select Countries"
              >
                {countryOptions}
              </Select>
            </Form.Item>
            <Form.Item
              label="Weeks to Predict"
              name="weeks"
              rules={[
                { required: true, message: "Please select number of weeks!" }
              ]}
            >
              <InputNumber min={1} defaultValue={3} />
            </Form.Item>
            <Form.Item>
              <Button type="primary" htmlType="submit">
                Predict!
              </Button>
            </Form.Item>
          </Form>
        </div>
        <div className="main-graph">
          <Graph data={mainGraphData}></Graph>
        </div>
      </div>
    );
  }
}

export default Covid19App;
