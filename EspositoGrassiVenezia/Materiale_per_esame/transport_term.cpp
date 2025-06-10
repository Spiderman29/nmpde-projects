class TransportTerm : public Function<dim>
  {
  public:
    // For vector-valued functions, it is good practice to define both the value
    // and the vector_value methods.
    virtual void
    vector_value(const Point<dim> & /*p*/,
                 Vector<double> &values) const override
    {
      values[0] = 0.0;
      values[1] = val;
    }

    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int /*component*/ = 0) const override
    {
        return val;
    }

  protected:
    const double val = -1.0;
  };